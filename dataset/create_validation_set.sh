#!/usr/bin/env bash

# Run it from this directory.

# 1st param - folder where images are found
# 2nd param - folder where annotations are found
# 3rd param - number of random choices
# 4th param - folder where validation images/annots end up (must have images/annots folders created)

ls "$1" | sort -R | tail -"$3" | while read image; do
   filename=$(basename "$image" .jpg)
   annot="$filename.xml"
   echo "moving files $image $annot"
   mv "$1/$image" "$4/images"
   mv "$2/$annot" "$4/annots"
done
