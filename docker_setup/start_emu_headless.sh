#!/bin/bash


# Credits to https://github.com/amrsa1/Android-Emulator-image

BL='\033[0;34m'
G='\033[0;32m'
RED='\033[0;31m'
YE='\033[1;33m'
NC='\033[0m' # No Color

emulator_name=${EMULATOR_NAME}

# Docker image layers are shared by all containers.  The emulator refuses to
# open the same AVD from multiple containers unless each worker has its own
# writable AVD directory.  Copy the initialized AVD once per container so app
# databases and userdata stay isolated across workers.
function isolate_avd() {
  if [[ "${ANDROID_WORLD_AVD_ISOLATE:-1}" != "1" ]]; then
    return
  fi

  local source_home="${ANDROID_AVD_HOME:-${HOME}/.android/avd}"
  local source_dir="${source_home}/${emulator_name}.avd"
  local isolated_home="/tmp/androidworld-avd-${HOSTNAME:-$$}"
  local isolated_name="${emulator_name}_${HOSTNAME:-$$}"
  local isolated_dir="${isolated_home}/${isolated_name}.avd"
  if [[ ! -d "$source_dir" ]]; then
    echo "Warning: AVD directory not found at ${source_dir}; using shared AVD." >&2
    return
  fi
  if [[ ! -d "$isolated_dir" ]]; then
    mkdir -p "$isolated_home"
    cp -a "$source_dir" "$isolated_dir"
    if [[ -f "${source_home}/${emulator_name}.ini" ]]; then
      cp "${source_home}/${emulator_name}.ini" "${isolated_home}/${isolated_name}.ini"
    fi
  fi
  if [[ -f "${isolated_home}/${isolated_name}.ini" ]]; then
    sed -i "s#^path=.*#path=${isolated_dir}#; s#^avd\.ini\.displayname=.*#avd.ini.displayname=${isolated_name}#" "${isolated_home}/${isolated_name}.ini"
  fi
  if [[ -f "${isolated_dir}/config.ini" ]]; then
    sed -i "s#^avd\.name *=.*#avd.name = ${isolated_name}#; s#^avd\.id *=.*#avd.id = ${isolated_name}#" "${isolated_dir}/config.ini"
  fi
  # A prepared image can contain this stale marker from the setup emulator.
  # Remove it before booting; otherwise the emulator reports a duplicate AVD.
  find "$isolated_dir" -maxdepth 1 -type f \( -name '*.lock' -o -name 'multiinstance.lock' \) -delete
  export ANDROID_AVD_HOME="$isolated_home"
  emulator_name="$isolated_name"
  export EMULATOR_NAME="$emulator_name"
  echo "Using isolated AVD home: ${ANDROID_AVD_HOME}"
  echo "Using isolated AVD name: ${EMULATOR_NAME}"
}

function check_hardware_acceleration() {
    if [[ "$HW_ACCEL_OVERRIDE" != "" ]]; then
        hw_accel_flag="$HW_ACCEL_OVERRIDE"
    else
        if [[ "$OSTYPE" == "darwin"* ]]; then
            # macOS-specific hardware acceleration check
            HW_ACCEL_SUPPORT=$(sysctl -a | grep -E -c '(vmx|svm)')
        else
            # generic Linux hardware acceleration check
            HW_ACCEL_SUPPORT=$(grep -E -c '(vmx|svm)' /proc/cpuinfo)
        fi

        if [[ $HW_ACCEL_SUPPORT == 0 ]]; then
            hw_accel_flag="-accel off"
            echo "Warning: no accelerator found. This Docker image is experimental and has only been tested on linux devices with KVM enabled."
        else
            hw_accel_flag="-accel on"
        fi
    fi

    echo "$hw_accel_flag"
}


hw_accel_flag=$(check_hardware_acceleration)

function launch_emulator () {
  isolate_avd
  if [[ "${ANDROID_WORLD_AVD_ISOLATE:-1}" != "1" ]]; then
    find "${ANDROID_AVD_HOME:-${HOME}/.android/avd}/${emulator_name}.avd" -maxdepth 1 -type f \( -name '*.lock' -o -name 'multiinstance.lock' \) -delete 2>/dev/null || true
  fi
  adb devices | grep emulator | cut -f1 | xargs -I {} adb -s "{}" emu kill
  # options="@${emulator_name} -no-window -no-snapshot -noaudio -no-boot-anim -memory 2048 ${hw_accel_flag} -camera-back none  -grpc 8554"
  options="@${emulator_name} -no-window -no-snapshot -no-boot-anim -noaudio -memory 2048 ${hw_accel_flag} -grpc 8554"
  if [[ "$OSTYPE" == *linux* ]]; then
    echo "${OSTYPE}: emulator ${options} -gpu off"
    nohup emulator $options -gpu off &
  fi
  if [[ "$OSTYPE" == *darwin* ]] || [[ "$OSTYPE" == *macos* ]]; then
    echo "${OSTYPE}: emulator ${options} -gpu swiftshader_indirect"
    nohup emulator $options -gpu swiftshader_indirect &
  fi

  if [ $? -ne 0 ]; then
    echo "Error launching emulator"
    return 1
  fi
}


function check_emulator_status () {
  printf "${G}==> ${BL}Checking emulator booting up status 🧐${NC}\n"
  start_time=$(date +%s)
  spinner=( "⠹" "⠺" "⠼" "⠶" "⠦" "⠧" "⠇" "⠏" )
  i=0
  # Get the timeout value from the environment variable or use the default value of 300 seconds (5 minutes)
  timeout=${EMULATOR_TIMEOUT:-300}

  while true; do
    result=$(adb shell getprop sys.boot_completed 2>&1)

    if [ "$result" == "1" ]; then
      printf "\e[K${G}==> \u2713 Emulator is ready : '$result'           ${NC}\n"
      adb devices -l
      adb shell input keyevent 82
      return 0  # Return a 0 to indicate emulator has booted successfully
    elif [ "$result" == "" ]; then
      printf "${YE}==> Emulator is partially Booted! 😕 ${spinner[$i]} ${NC}\r"
    else
      printf "${RED}==> $result, please wait ${spinner[$i]} ${NC}\r"
      i=$(( (i+1) % 8 ))
    fi

    current_time=$(date +%s)
    elapsed_time=$((current_time - start_time))
    if [ $elapsed_time -gt $timeout ]; then
      printf "${RED}==> Timeout after ${timeout} seconds elapsed 🕛.. ${NC}\n"
      return 1 # Return a 1 to indicate failure if exceeded timeout
    fi
    sleep 4
  done
};


function disable_animation() {
  adb shell "settings put global window_animation_scale 0.0"
  adb shell "settings put global transition_animation_scale 0.0"
  adb shell "settings put global animator_duration_scale 0.0"
};

function hidden_policy() {
  adb shell "settings put global hidden_api_policy_pre_p_apps 1;settings put global hidden_api_policy_p_apps 1;settings put global hidden_api_policy 1"
};

launch_emulator
sleep 2

if check_emulator_status; then
  # Only run the below if the emulator is actually ready
  sleep 1
  disable_animation
  sleep 1
  hidden_policy
  sleep 1
else
  echo "Emulator failed to start properly, exiting..."
  exit 1
fi
