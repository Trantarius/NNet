 
g++ -w -std=c++17 tools/$1.cpp -lpng -o $1
echo
echo compiled
echo
./$1
rm $1
