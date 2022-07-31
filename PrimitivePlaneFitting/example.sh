python preProcess.py --target ply
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j8
cd ..
./bin/Release/region_growing_3d
python preProcess.py --target npz


