# convert jpg to png
cd ../input
mogrify -format png32 *.jpg
for x in *.png32; do mv "$x" "${x%.png32}.png"; done
mv *.png ../input-png

# extract mask
cd ../mask
for x in *.jpg; do convert "$x" -background none -transparent black -flatten -morphology Dilate Octagon  "${x%.jpg}.png" ; done
mv *.png ../input-png
cd ../input-png

# merge and remove files
for x in cat.*.png; do
  convert $x mask_$x \
          -alpha Off -compose CopyOpacity -composite -trim +repage \
          $x
  rm mask_$x
done
