#!/bin/bash
rm salida.txt
./PedestrainNavigation_visualisation.sh >> salida.txt & ./tiempo.sh >> salida.txt
