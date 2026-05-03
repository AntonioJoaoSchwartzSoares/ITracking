@echo off
REM Chama docker-run.ps1 sem depender da ExecutionPolicy (Bypass só neste processo).
REM Uso (na raiz do repo ou noutra pasta):
REM   scripts\docker-run.cmd -UseOriginal -Video ".\data\input\VISTA_AEREA.mp4" -Out ".\data\out" --conf 0.2
REM   (Só use --class-id 2 se souber que o kart entra como "car" no COCO; senão omita ou use --class-id -1.)

set "HERE=%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%HERE%docker-run.ps1" %*
