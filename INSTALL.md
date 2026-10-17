# Audio Repair: installing the desktop app

Audio Repair fixes clicks, bumps, clipping, background noise, room echo, poor mic tone and uneven loudness in
recordings. It is a normal desktop app: it opens in its own window, works offline, and needs nothing else installed
(no Python, no scripts, no browser).

## Which file to download

| Your computer | File |
|---|---|
| Windows 10 / 11 (normal PC) | `Audio-Repair-Setup-1.0.0-x64.exe` (installer, recommended) or `Audio-Repair-Portable-1.0.0-x64.exe` (runs without installing; slower to open) |
| Mac with Apple silicon (M1, M2, M3, M4...) | `Audio-Repair-1.0.0-mac-arm64.dmg` |
| Linux (normal PC) | `Audio-Repair-1.0.0-linux-x86_64.AppImage` (any distro) or the `.tar.gz` |

Intel Macs and Windows on ARM are not built yet.

## Installing

- **Windows installer:** double-click it, then Next, then Install. It installs just for you (no administrator
  password), and adds Audio Repair to the Start menu and, if you tick it, the desktop. Remove it any time from
  Settings, Apps, Installed apps, Audio Repair, Uninstall.
- **Windows portable:** put the file anywhere and double-click it. Nothing is installed.
- **Mac:** open the `.dmg` and drag **Audio Repair** onto **Applications**.
- **Linux:** make the AppImage executable (`chmod +x Audio-Repair-*.AppImage`, or Properties, Permissions,
  "Allow executing"), then double-click it.

## First launch: security warnings

The app is not signed with a paid certificate, so your system warns once:

- **Windows:** "Windows protected your PC" (SmartScreen): click **More info**, then **Run anyway**.
- **Mac:** "cannot be opened because the developer cannot be verified": in Finder, **right-click** the app, choose
  **Open**, then **Open**. On macOS 15 or newer: try to open it once, then System Settings, Privacy & Security,
  **Open Anyway**.

## Using it

1. Open **Audio Repair**.
2. Click **Choose file** (or drop a recording onto the window). WAV, MP3, FLAC, OGG, M4A and MP4 voice memos work.
3. Leave the settings on their defaults, or switch repairs on and off.
4. Click **Repair audio**. Expect roughly half a minute of work per minute of audio (measured: a 109-second voice
   memo took about a minute).
5. Compare **Before** and **After**, read **What was done**, then click **Save repaired file** and choose where to
   save it. Your original is never changed.

## Good to know

- Everything happens on your computer. Recordings are never uploaded anywhere, and the app opens no network
  connection.
- Working copies are kept in your temporary folder and cleaned up automatically after an hour.
- The installed apps use the built-in noise remover. The extra neural noise remover (DeepFilterNet) needs a
  developer setup and is only used when running from the source folder.
