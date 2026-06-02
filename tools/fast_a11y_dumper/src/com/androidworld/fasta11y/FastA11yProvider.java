package com.androidworld.fasta11y;

import android.content.ContentProvider;
import android.content.ContentValues;
import android.database.Cursor;
import android.net.Uri;
import android.os.Bundle;
import android.os.CancellationSignal;
import android.os.ParcelFileDescriptor;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.charset.StandardCharsets;

public final class FastA11yProvider extends ContentProvider {
  private static final String MIME_JSON = "application/json";

  @Override
  public boolean onCreate() {
    return true;
  }

  @Override
  public String getType(Uri uri) {
    return MIME_JSON;
  }

  @Override
  public ParcelFileDescriptor openFile(Uri uri, String mode) throws FileNotFoundException {
    if (!"r".equals(mode)) {
      throw new FileNotFoundException("FastA11yProvider is read-only");
    }
    return openPipeHelper(
        uri,
        MIME_JSON,
        null,
        null,
        new PipeDataWriter<Object>() {
          @Override
          public void writeDataToPipe(
              ParcelFileDescriptor output, Uri uri, String mimeType, Bundle opts, Object args) {
            writeSnapshot(output, uri, mimeType, opts, args);
          }
        });
  }

  @Override
  public ParcelFileDescriptor openFile(Uri uri, String mode, CancellationSignal signal)
      throws FileNotFoundException {
    return openFile(uri, mode);
  }

  private void writeSnapshot(ParcelFileDescriptor output, Uri uri, String mimeType, Bundle opts, Object args) {
    byte[] payload;
    try {
      FastA11yService service = FastA11yService.getInstance();
      if (service == null) {
        payload = FastA11yService.notReadyJson().getBytes(StandardCharsets.UTF_8);
      } else {
        boolean flat = "flat".equals(uri.getLastPathSegment());
        boolean compact = "1".equals(uri.getQueryParameter("compact"))
            || "true".equals(uri.getQueryParameter("compact"));
        int maxNodes = parsePositiveInt(uri.getQueryParameter("max_nodes"), 10000);
        payload = service.snapshotJson(flat, compact, maxNodes).getBytes(StandardCharsets.UTF_8);
      }
    } catch (Throwable t) {
      payload = FastA11yService.errorJson(t).getBytes(StandardCharsets.UTF_8);
    }

    try (FileOutputStream stream = new FileOutputStream(output.getFileDescriptor())) {
      stream.write(payload);
    } catch (IOException ignored) {
      // The caller may close the adb content stream early.
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

  @Override
  public Cursor query(
      Uri uri,
      String[] projection,
      String selection,
      String[] selectionArgs,
      String sortOrder) {
    return null;
  }

  @Override
  public Uri insert(Uri uri, ContentValues values) {
    return null;
  }

  @Override
  public int delete(Uri uri, String selection, String[] selectionArgs) {
    return 0;
  }

  @Override
  public int update(Uri uri, ContentValues values, String selection, String[] selectionArgs) {
    return 0;
  }
}
