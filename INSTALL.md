# Audio Repair: how to install

Audio Repair fixes clicks, bumps, clipping, background noise, room echo, poor mic tone and uneven loudness in
recordings. It is a normal desktop app: it opens in its own window, works offline, and needs nothing else installed
(no Python, no scripts, no browser).

## Which file

| Your computer | File |
|---|---|
| Windows 10 / 11 | `Audio Repair - Windows Installer.exe` |
| Mac with Apple silicon (M1, M2, M3, M4...) | `Audio Repair - Mac Installer.dmg` |

Intel Macs and Windows on ARM are not built.

## Installing

- **Windows:** double-click the installer, then Next, then Install. It installs just for you (no administrator
  password) and adds Audio Repair to the Start menu and, if you tick it, the desktop. Remove it any time from
  Settings, Apps, Installed apps, Audio Repair, Uninstall.
- **Mac:** open the `.dmg` and drag **Audio Repair** onto **Applications**.

## First launch: security warning

The app is not signed with a paid certificate, so your computer warns once:

- **Windows:** "Windows protected your PC": click **More info**, then **Run anyway**.
- **Mac:** "cannot be opened because the developer cannot be verified": in Finder, **right-click** the app, choose
  **Open**, then **Open**. On macOS 15 or newer: try to open it once, then System Settings, Privacy & Security,
  **Open Anyway**.

## Using it

1. Open **Audio Repair** (Start menu or desktop shortcut on Windows, Applications on a Mac).
2. Click **Choose file** (or drop a recording onto the window). WAV, MP3, FLAC, OGG, M4A and MP4 voice memos work.
3. Leave the settings on their defaults, or switch repairs on and off.
4. Click **Repair audio**. A 13-second clip takes about 12 seconds; a 2-minute voice memo about a minute.
5. Compare **Before** and **After**. Drag across either picture to scrub, and switch **Picture** to **Spectrum** to
   see the sound as colours (brighter = louder, higher = higher pitch). Before and After stay at the same position, so
   you can switch between them while listening.
6. Change the **Tone** if you want: **Warmth**, **Clarity**, **Brightness** and **Air**, or the Warmer / Brighter /
   Clearer buttons. You hear it straight away, and it never repairs the recording again.
7. Read **What was done**, then click **Save** and choose where to save it (with your tone changes, if you made any).
   Your original is never changed.

## Updates

Updates come as one file named like `Audio Repair - Mk II update.srar`. To install one:

1. Double-click the `.srar` file (or drop it onto the Audio Repair window, or scroll to **Updates** in the app and
   click **Choose…**).
2. Read what the update changes, then click **Install**, then **Restart now**.

**Updates** at the bottom of the app shows which version you are using (Mk I, Mk II, ...). Earlier versions stay
installed: click **Use** next to one to go back to it. An update never changes your recordings or the installed
program; you can uninstall the app as usual at any time.

## Good to know

- Everything happens on your computer. Recordings are never uploaded anywhere.
- The AI models come with the app and work offline; nothing is downloaded later:
  - **DeepFilterNet3** removes background noise and room echo from voices. It is always used for voice recordings.
  - **Studio voice (AI)** (Resemble Enhance) rebuilds the whole voice. It is off by default: switch it on under
    Repairs. It is slow (roughly a minute and a half per 10 seconds of audio) and can sound less natural than the
    normal repair, so compare Before and After and keep whichever you prefer.
- Because the AI models are included, the installers are large (Windows about 860 MB, Mac about 950 MB) and the
  installed app takes about 1.5 GB of disk space.
