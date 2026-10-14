; Inno Setup Script for AI Gesture Controller (Microsoft Store Ready)
[Setup]
AppId={{8B41D89F-839D-42A9-9C55-6F479C1B2A10}
AppName=AI Gesture Controller
AppVersion=1.0.0
AppPublisher=AI Gesture Assistant
AppPublisherURL=https://github.com
AppSupportURL=https://github.com
AppUpdatesURL=https://github.com
DefaultDirName={autopf}\AIGestureController
DefaultGroupName=AI Gesture Controller
DisableProgramGroupPage=yes
LicenseFile=
OutputDir=d:\DL_PROJECT\dist_installer
OutputBaseFilename=AIGestureController_Setup
SetupIconFile=d:\DL_PROJECT\app_icon.ico
Compression=lzma2/ultra64
SolidCompression=yes
WizardStyle=modern
PrivilegesRequired=lowest
ArchitecturesInstallIn64BitMode=x64compatible
UninstallDisplayIcon={app}\AIGestureController.exe

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
Source: "d:\DL_PROJECT\dist\AIGestureController\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{autoprograms}\AI Gesture Controller"; Filename: "{app}\AIGestureController.exe"; IconFilename: "{app}\AIGestureController.exe"
Name: "{autodesktop}\AI Gesture Controller"; Filename: "{app}\AIGestureController.exe"; IconFilename: "{app}\AIGestureController.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\AIGestureController.exe"; Description: "{cm:LaunchProgram,AI Gesture Controller}"; Flags: nowait postinstall skipifsilent
