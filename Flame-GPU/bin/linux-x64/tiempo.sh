#!/bin/bash
segundos=0
while sleep 0.$((1999999999 - 1$(date +%N)))
 do
   let "segundos++" 
   echo Pasaron $segundos segundos
 done
