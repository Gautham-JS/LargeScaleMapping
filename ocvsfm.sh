echo "Enter fname"
read fname
echo "Compiling $fname.cxx"
g++ $fname -o app `pkg-config --cflags --libs opencv`
echo "Enter Args[3]"
read f
read cx
read cy
echo "Building ./$fname with args $f,$cx,$cy"
./app aths.txt $f $cx $cy

