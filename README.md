# PCGSDF

===

Procedural Content Generation with Signed Distance Functions.

Still in development, initially intended for point cloud generation for testing of point cloud registration algorithms.

## Build

```
mkdir -p build
pushd build
cmake ../ -DCMAKE_BUILD_TYPE=RelWithDebInfo
make -j$(nproc)
popd
```

## Run example

Command to run preliminary example:

```
pushd build
./pcgsdf/gen/sdf/test/cho_gen_sdf_test
popd
```

### Example output

![demo](img/2021-01-04-output.gif)

```
$ build/pcgsdf/gen/sdf/test/cho_gen_sdf_test

seed=1609792766012069439
Scene Generation Start.
Scene Generation End.
Trajectory Generation Start.
Gen took 445 ms
Trajectory Generation End : success.
```
