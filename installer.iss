; Windows installer for Audio Repair (Inno Setup 6).
; Built by .github/workflows/build.yml after build.py has produced dist\Audio Repair\.
; Installs per user (no administrator rights needed), adds Start-menu and desktop shortcuts and an uninstaller,
; registers .srar update files (double-click one to install the update) and puts HANDOFF.md next to the program.

#define AppName "Audio Repair"
#ifndef AppVersion
  #define AppVersion "1.1.0"
#endif

[Setup]
AppId={{6F1C3E2A-8B4D-4E7A-9C51-2D7A0B3E9F14}
AppName={#AppName}
AppVersion={#AppVersion}
AppVerName={#AppName} {#AppVersion}
AppPublisher=Audio Repair
DefaultDirName={localappdata}\Programs\{#AppName}
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
OutputDir=.
OutputBaseFilename=Audio Repair - Windows Installer
SetupIconFile=assets\icon.ico
UninstallDisplayIcon={app}\{#AppName}.exe
UninstallDisplayName={#AppName}
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
ChangesAssociations=yes
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Create a &desktop shortcut"; GroupDescription: "Shortcuts:"

[Files]
Source: "dist\{#AppName}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
; how the app is made (for Claude or a developer); the app replaces it with the newer copy each update carries
Source: "HANDOFF.md"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{group}\{#AppName}"; Filename: "{app}\{#AppName}.exe"
Name: "{group}\Uninstall {#AppName}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#AppName}.exe"; Tasks: desktopicon

[Registry]
; .srar = an Audio Repair update; per-user classes (HKA is HKCU for a per-user install)
Root: HKA; Subkey: "Software\Classes\.srar"; ValueType: string; ValueName: ""; ValueData: "AudioRepair.Update"; Flags: uninsdeletevalue
Root: HKA; Subkey: "Software\Classes\AudioRepair.Update"; ValueType: string; ValueName: ""; ValueData: "Audio Repair update"; Flags: uninsdeletekey
Root: HKA; Subkey: "Software\Classes\AudioRepair.Update\DefaultIcon"; ValueType: string; ValueName: ""; ValueData: "{app}\{#AppName}.exe,0"
Root: HKA; Subkey: "Software\Classes\AudioRepair.Update\shell\open\command"; ValueType: string; ValueName: ""; ValueData: """{app}\{#AppName}.exe"" ""%1"""

[Run]
Filename: "{app}\{#AppName}.exe"; Description: "Open {#AppName} now"; Flags: nowait postinstall skipifsilent
