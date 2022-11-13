#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "Don't forget directory name!"
    exit
fi


for file in $(ls $1); do
	echo $file
  fpath=$1/$file
  perl -i -p -e 's/,\n/, /g' $fpath
  perl -i -p -e 's/^(EToE|EToV|EToF|EToB|mesh\_Vx|mesh\_Vy|mesh\_DGNx|mesh\_DGNy)[.\s=\[0-9\-,e\]]*$//gm' $fpath
  perl -i -p -e 's/\], /\],\n/g' $fpath
done;

