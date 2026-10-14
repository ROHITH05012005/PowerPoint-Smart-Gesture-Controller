Dim fso, WshShell, currentDir
Set WshShell = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")

' Get directory of this VBS script
currentDir = fso.GetParentFolderName(WScript.ScriptFullName)
WshShell.CurrentDirectory = currentDir

' Launch AI Gesture Controller seamlessly without terminal background
WshShell.Run "pythonw.exe main.py", 0, False
