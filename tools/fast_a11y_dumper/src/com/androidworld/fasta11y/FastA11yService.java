package com.androidworld.fasta11y;

import android.accessibilityservice.AccessibilityService;
import android.graphics.Rect;
import android.net.LocalServerSocket;
import android.net.LocalSocket;
import android.os.Build;
import android.view.accessibility.AccessibilityEvent;
import android.view.accessibility.AccessibilityNodeInfo;
import android.view.accessibility.AccessibilityWindowInfo;
import java.io.BufferedReader;
import java.io.DataOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.List;

public final class FastA11yService extends AccessibilityService {
  private static final String SOCKET_NAME = "androidworld_fast_a11y";
  private static volatile FastA11yService instance;
  private volatile boolean socketRunning;
  private LocalServerSocket serverSocket;

  public static FastA11yService getInstance() {
    return instance;
  }

  public static String notReadyJson() {
    return "{\"ok\":false,\"error\":\"accessibility_service_not_connected\"}\n";
  }

  public static String errorJson(Throwable t) {
    StringBuilder out = new StringBuilder(256);
    out.append("{\"ok\":false,\"error\":\"");
    appendJsonStringContent(out, t.getClass().getSimpleName() + ": " + String.valueOf(t.getMessage()));
    out.append("\"}\n");
    return out.toString();
  }

  @Override
  protected void onServiceConnected() {
    instance = this;
    startSocketServer();
  }

  @Override
  public void onDestroy() {
    stopSocketServer();
    if (instance == this) {
      instance = null;
    }
    super.onDestroy();
  }

  @Override
  public void onAccessibilityEvent(AccessibilityEvent event) {
    // Tree retrieval is pull-based through FastA11yProvider.
  }

  @Override
  public void onInterrupt() {
    // No ongoing spoken/audio feedback to interrupt.
  }

  private synchronized void startSocketServer() {
    if (socketRunning) {
      return;
    }
    socketRunning = true;
    Thread thread = new Thread(
        new Runnable() {
          @Override
          public void run() {
            runSocketServer();
          }
        },
        "FastA11ySocketServer");
    thread.setDaemon(true);
    thread.start();
  }

  private void runSocketServer() {
    try {
      serverSocket = new LocalServerSocket(SOCKET_NAME);
      while (socketRunning) {
        final LocalSocket client = serverSocket.accept();
        Thread handler = new Thread(
            new Runnable() {
              @Override
              public void run() {
                handleSocketClient(client);
              }
            },
            "FastA11ySocketClient");
        handler.setDaemon(true);
        handler.start();
      }
    } catch (Throwable ignored) {
      // Closing the server during service shutdown also exits through here.
    } finally {
      socketRunning = false;
    }
  }

  private void handleSocketClient(LocalSocket client) {
    try {
      BufferedReader input = new BufferedReader(
          new InputStreamReader(client.getInputStream(), StandardCharsets.UTF_8));
      DataOutputStream output = new DataOutputStream(client.getOutputStream());
      String request;
      while (socketRunning && (request = input.readLine()) != null) {
        String[] fields = request.trim().split("\\s+");
        boolean flat = fields.length < 1 || !"tree".equals(fields[0]);
        boolean compact = fields.length >= 2 && "1".equals(fields[1]);
        int maxNodes = fields.length >= 3 ? parsePositiveInt(fields[2], 10000) : 10000;
        byte[] payload = snapshotJson(flat, compact, maxNodes)
            .getBytes(StandardCharsets.UTF_8);
        output.writeInt(payload.length);
        output.write(payload);
        output.flush();
      }
    } catch (Throwable ignored) {
      // A host-side abort intentionally closes the forwarded connection.
    } finally {
      try {
        client.close();
      } catch (IOException ignored) {
        // Nothing else to release.
      }
    }
  }

  private synchronized void stopSocketServer() {
    socketRunning = false;
    if (serverSocket != null) {
      try {
        serverSocket.close();
      } catch (IOException ignored) {
        // Nothing else to release.
      }
      serverSocket = null;
    }
  }

  private static int parsePositiveInt(String value, int fallback) {
    if (value == null || value.isEmpty()) {
      return fallback;
    }
    try {
      int parsed = Integer.parseInt(value);
      return parsed > 0 ? parsed : fallback;
    } catch (NumberFormatException e) {
      return fallback;
    }
  }

  public String snapshotJson(boolean flat, boolean compact, int maxNodes) {
    long startedNs = System.nanoTime();
    List<AccessibilityWindowInfo> windows = getWindows();
    AccessibilityNodeInfo activeRoot = null;
    if (windows == null || windows.isEmpty()) {
      activeRoot = getRootInActiveWindow();
    }
    long capturedNs = System.nanoTime();

    SnapshotWriter writer = new SnapshotWriter(flat, compact, maxNodes);
    String body = writer.write(windows, activeRoot);
    long serializedNs = System.nanoTime();

    StringBuilder out = new StringBuilder(body.length() + 256);
    out.append("{\"ok\":true");
    out.append(",\"format\":\"").append(flat ? "flat" : "tree").append("\"");
    out.append(",\"compact\":").append(compact);
    out.append(",\"captureMs\":").append(ms(capturedNs - startedNs));
    out.append(",\"serializeMs\":").append(ms(serializedNs - capturedNs));
    out.append(",\"serviceMs\":").append(ms(serializedNs - startedNs));
    out.append(",\"nodeCount\":").append(writer.nodeCount);
    out.append(",\"emittedCount\":").append(writer.emittedCount);
    out.append(",\"truncated\":").append(writer.truncated);
    out.append(",\"payloadBytes\":").append(body.getBytes(java.nio.charset.StandardCharsets.UTF_8).length);
    out.append(',');
    out.append(body);
    out.append("}\n");
    return out.toString();
  }

  private static String ms(long nanos) {
    return String.format(java.util.Locale.US, "%.3f", nanos / 1_000_000.0);
  }

  private static final class SnapshotWriter {
    private final boolean flat;
    private final boolean compact;
    private final int maxNodes;
    private final StringBuilder out = new StringBuilder(64 * 1024);
    private final Rect bounds = new Rect();
    int nodeCount = 0;
    int emittedCount = 0;
    boolean truncated = false;

    SnapshotWriter(boolean flat, boolean compact, int maxNodes) {
      this.flat = flat;
      this.compact = compact;
      this.maxNodes = maxNodes;
    }

    String write(List<AccessibilityWindowInfo> windows, AccessibilityNodeInfo activeRoot) {
      if (flat) {
        out.append("\"nodes\":[");
      } else {
        out.append("\"windows\":[");
      }

      boolean[] first = new boolean[] {true};
      if (windows != null && !windows.isEmpty()) {
        for (int i = 0; i < windows.size(); i++) {
          AccessibilityWindowInfo window = windows.get(i);
          if (flat) {
            appendFlatWindow(window, i, first);
          } else {
            appendTreeWindow(window, i, first);
          }
          if (truncated) {
            break;
          }
        }
      } else if (activeRoot != null) {
        if (flat) {
          appendFlatNode(activeRoot, -1, 0, first, false);
        } else {
          appendTreeRoot(activeRoot, first);
        }
      }

      out.append(']');
      return out.toString();
    }

    private void appendTreeWindow(AccessibilityWindowInfo window, int windowIndex, boolean[] first) {
      if (!first[0]) {
        out.append(',');
      }
      first[0] = false;
      out.append('{');
      out.append("\"windowIndex\":").append(windowIndex);
      out.append(",\"type\":").append(window.getType());
      out.append(",\"layer\":").append(window.getLayer());
      out.append(",\"active\":").append(window.isActive());
      out.append(",\"focused\":").append(window.isFocused());
      out.append(",\"root\":");
      AccessibilityNodeInfo root = window.getRoot();
      if (root == null) {
        out.append("null");
      } else {
        appendTreeNode(root, 0);
      }
      out.append('}');
    }

    private void appendTreeRoot(AccessibilityNodeInfo root, boolean[] first) {
      if (!first[0]) {
        out.append(',');
      }
      first[0] = false;
      out.append("{\"windowIndex\":-1,\"root\":");
      appendTreeNode(root, 0);
      out.append('}');
    }

    private void appendTreeNode(AccessibilityNodeInfo node, int depth) {
      if (!beginNode(node)) {
        out.append("null");
        return;
      }
      out.append('{');
      appendNodeFields(node, depth, -1, false);
      out.append(",\"children\":[");
      boolean firstChild = true;
      int childCount = node.getChildCount();
      for (int i = 0; i < childCount; i++) {
        AccessibilityNodeInfo child = node.getChild(i);
        if (child == null) {
          continue;
        }
        if (!firstChild) {
          out.append(',');
        }
        firstChild = false;
        appendTreeNode(child, depth + 1);
        if (truncated) {
          break;
        }
      }
      out.append("]}");
    }

    private void appendFlatWindow(AccessibilityWindowInfo window, int windowIndex, boolean[] first) {
      AccessibilityNodeInfo root = window.getRoot();
      if (root != null) {
        appendFlatNode(root, windowIndex, 0, first, false);
      }
    }

    // Standard Material Design widget used to render a navigation drawer's
    // own menu content, not a per-app naming convention. Checked on the
    // widget's actual class name (same category of signal as the existing
    // role checks for switch/checkbox/button elsewhere in this pipeline),
    // not on any label or resource-id text. Deliberately does NOT match
    // DrawerLayout: that is the whole-screen root container hosting BOTH
    // the drawer and the main content side by side, so tagging everything
    // under it would mark essentially the entire screen as "drawer" on any
    // app using the standard drawer pattern - only NavigationView is
    // actually scoped to the drawer's own menu items.
    private static boolean isDrawerContainerClass(CharSequence className) {
      if (className == null) {
        return false;
      }
      String value = className.toString();
      return value.contains("NavigationView");
    }

    private void appendFlatNode(
        AccessibilityNodeInfo node, int windowIndex, int depth, boolean[] first,
        boolean inNavigationDrawer) {
      if (!beginNode(node)) {
        return;
      }
      boolean childInNavigationDrawer = inNavigationDrawer || isDrawerContainerClass(node.getClassName());
      boolean emit = !compact || isInteresting(node);
      if (emit) {
        if (!first[0]) {
          out.append(',');
        }
        first[0] = false;
        out.append('{');
        appendNodeFields(node, depth, windowIndex, true);
        out.append(",\"inNavigationDrawer\":").append(inNavigationDrawer);
        out.append('}');
        emittedCount++;
      }
      int childCount = node.getChildCount();
      for (int i = 0; i < childCount; i++) {
        AccessibilityNodeInfo child = node.getChild(i);
        if (child != null) {
          appendFlatNode(child, windowIndex, depth + 1, first, childInNavigationDrawer);
          if (truncated) {
            break;
          }
        }
      }
    }

    private boolean beginNode(AccessibilityNodeInfo node) {
      if (nodeCount >= maxNodes) {
        truncated = true;
        return false;
      }
      nodeCount++;
      if (!flat) {
        emittedCount++;
      }
      return true;
    }

    private static boolean isInteresting(AccessibilityNodeInfo node) {
      return hasText(node.getText())
          || hasText(node.getContentDescription())
          || hasText(node.getViewIdResourceName())
          || node.isClickable()
          || node.isLongClickable()
          || node.isFocusable()
          || node.isEditable()
          || node.isScrollable()
          || node.isCheckable();
    }

    private static boolean hasText(CharSequence value) {
      return value != null && value.length() > 0;
    }

    private void appendNodeFields(
        AccessibilityNodeInfo node, int depth, int windowIndex, boolean includeWindowIndex) {
      out.append("\"id\":").append(nodeCount - 1);
      if (includeWindowIndex) {
        out.append(",\"windowIndex\":").append(windowIndex);
      }
      out.append(",\"depth\":").append(depth);
      appendStringField("class", node.getClassName());
      appendStringField("package", node.getPackageName());
      appendStringField("resourceId", node.getViewIdResourceName());
      appendStringField("text", node.getText());
      appendStringField("contentDescription", node.getContentDescription());
      if (Build.VERSION.SDK_INT >= 26) {
        appendStringField("hint", node.getHintText());
      }
      if (Build.VERSION.SDK_INT >= 30) {
        appendStringField("stateDescription", node.getStateDescription());
      }
      node.getBoundsInScreen(bounds);
      out.append(",\"bounds\":[")
          .append(bounds.left)
          .append(',')
          .append(bounds.top)
          .append(',')
          .append(bounds.right)
          .append(',')
          .append(bounds.bottom)
          .append(']');
      out.append(",\"enabled\":").append(node.isEnabled());
      out.append(",\"visible\":").append(node.isVisibleToUser());
      out.append(",\"clickable\":").append(node.isClickable());
      out.append(",\"longClickable\":").append(node.isLongClickable());
      out.append(",\"focusable\":").append(node.isFocusable());
      out.append(",\"focused\":").append(node.isFocused());
      out.append(",\"editable\":").append(node.isEditable());
      out.append(",\"scrollable\":").append(node.isScrollable());
      out.append(",\"checkable\":").append(node.isCheckable());
      out.append(",\"checked\":").append(node.isChecked());
      out.append(",\"selected\":").append(node.isSelected());
      out.append(",\"childCount\":").append(node.getChildCount());
    }

    private void appendStringField(String name, CharSequence value) {
      if (value == null || value.length() == 0) {
        return;
      }
      out.append(",\"");
      out.append(name);
      out.append("\":\"");
      appendJsonStringContent(out, value.toString());
      out.append('"');
    }
  }

  private static void appendJsonStringContent(StringBuilder out, String value) {
    for (int i = 0; i < value.length(); i++) {
      char c = value.charAt(i);
      switch (c) {
        case '"':
          out.append("\\\"");
          break;
        case '\\':
          out.append("\\\\");
          break;
        case '\b':
          out.append("\\b");
          break;
        case '\f':
          out.append("\\f");
          break;
        case '\n':
          out.append("\\n");
          break;
        case '\r':
          out.append("\\r");
          break;
        case '\t':
          out.append("\\t");
          break;
        default:
          if (c < 0x20) {
            out.append(String.format(java.util.Locale.US, "\\u%04x", (int) c));
          } else {
            out.append(c);
          }
      }
    }
  }
}
