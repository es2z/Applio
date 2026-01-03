# ASIO Support for Applio Realtime Voice Conversion

This directory is for placing a custom PortAudio DLL with ASIO support.

## Why is this needed?

The standard `sounddevice` Python package does not include ASIO support due to licensing restrictions. To use ASIO audio devices (like MOTU M Series, Focusrite, RME, etc.) in Applio's realtime mode, you need to provide an ASIO-enabled PortAudio DLL.

## How to enable ASIO support

### Option 1: Download pre-built DLL (Recommended)

**Download from [spatialaudio/portaudio-binaries](https://github.com/spatialaudio/portaudio-binaries):**

1. Go to: https://github.com/spatialaudio/portaudio-binaries
2. Download `libportaudio64bit-asio.dll` (click the file, then "Download raw file")
   - Direct link: https://github.com/spatialaudio/portaudio-binaries/raw/master/libportaudio64bit-asio.dll
3. Place it in this directory (`assets/portaudio/`) - no need to rename!
4. Restart Applio

**Supported DLL filenames:**
- `libportaudio64bit-asio.dll` (from spatialaudio/portaudio-binaries)
- `portaudio_x64.dll` (if you build it yourself)

### Option 2: Build from source

#### Prerequisites
- CMake 3.18 or later
- Visual Studio 2019/2022 (or Build Tools)
- Git

#### Steps

1. **Download ASIO SDK** (now open source under GPLv3):
   - Official: https://www.steinberg.net/developers/
   - Direct link: https://download.steinberg.net/sdk_downloads/ASIO-SDK_2.3.4_2025-10-15.zip

2. **Clone PortAudio**:
   ```cmd
   git clone https://github.com/PortAudio/portaudio.git
   cd portaudio
   ```

3. **Extract ASIO SDK**:
   - Extract the ASIO SDK to a known location (e.g., `C:\ASIOSDK`)

4. **Build with CMake**:
   ```cmd
   mkdir build
   cd build
   cmake .. -G "Visual Studio 17 2022" -A x64 -DPA_USE_ASIO=ON -DASIO_SDK_PATH=C:\ASIOSDK
   cmake --build . --config Release
   ```

5. **Copy the DLL**:
   - Find `portaudio_x64.dll` in the build output directory
   - Copy it to `assets/portaudio/portaudio_x64.dll`

6. **Restart Applio**

## Verification

After placing the DLL and restarting Applio:

1. Go to the **Realtime** tab
2. Click **Refresh Devices**
3. ASIO devices should now appear in the device list with "(ASIO)" suffix

Example: `Loopback (MOTU M Series) (ASIO)`

## Troubleshooting

### ASIO devices still not showing
- Ensure the DLL is named exactly `portaudio_x64.dll`
- Check the console for `[PortAudio] Loaded custom DLL with ASIO support` message
- Make sure your ASIO driver is properly installed

### Device in use error
- ASIO requires exclusive access
- Close other applications using the ASIO device (DAW, etc.)
- Or use ASIO4ALL to share the device

### Mixed API usage (ASIO input + WASAPI output)
- This is supported - Applio will automatically use separate streams
- Look for `[ASIO mixed mode]` in the console log

## License

- ASIO SDK: GPLv3 (as of October 2025) or Steinberg proprietary license
- PortAudio: MIT license
- This DLL is not included in the repository due to licensing considerations
