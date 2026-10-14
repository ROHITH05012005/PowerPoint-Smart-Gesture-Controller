$desktopPath = [Environment]::GetFolderPath("Desktop")
$exePath = "d:\DL_PROJECT\dist\AIGestureController\AIGestureController.exe"
$workingDir = "d:\DL_PROJECT\dist\AIGestureController"
$iconPath = "d:\DL_PROJECT\app_icon.ico"
$shortcutPath = "$desktopPath\AI Gesture Controller.lnk"

$WshShell = New-Object -ComObject WScript.Shell
$Shortcut = $WshShell.CreateShortcut($shortcutPath)
$Shortcut.TargetPath = $exePath
$Shortcut.WorkingDirectory = $workingDir
$Shortcut.IconLocation = "$iconPath, 0"
$Shortcut.Description = "AI-Powered PowerPoint Smart Gesture Controller"
$Shortcut.Save()

Write-Host "SUCCESS: Desktop shortcut now links directly to $exePath"
