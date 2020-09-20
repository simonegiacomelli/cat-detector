mogrify -format png32 *.jpg
for x in *.png32; do mv "$x" "${x%.png32}.png"; done
mv *.png ../input-png