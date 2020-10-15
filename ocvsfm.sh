echo "Enter fname"
read fname
echo "Compiling $fname.cxx"

g++ $fname.cpp -o app `pkg-config --cflags --libs opencv`
if [ "$?" = "0" ]; then
  echo "Compiled"
else
  echo "Terminate"
  exit 1
fi

echo "Enter Args focal length, Cx, Cy"
read f
read cx
read cy
echo "Building ./$fname with args $f,$cx,$cy"

./app aths.txt $f $cx $cy
if [ "$?" = "0" ]; then
  echo " "
else
  echo "Terminate with exit 1"
  exit 1
fi
