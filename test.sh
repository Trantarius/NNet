 
g++ -w -O3 -std=c++17 $(find src -name "*.cpp") -lpng -o exe
echo
echo compiled
echo
./exe
