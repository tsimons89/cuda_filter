import os

# cuda_filter_src = Split('''src/Eco_filter.cpp src/Eco_file_access.cpp ''')

opencv_libs = Split('''opencv_core opencv_highgui opencv_imgproc opencv_objdetect opencv_ml opencv_imgcodecs opencv_videoio''')

# cuda_opencv_libs = Split('''opencv_cudaarithm opencv_cudabgsegm opencv_cudacodec opencv_cudafeatures2d opencv_cudafilters 
# 	opencv_cudaimgproc opencv_cudalegacy opencv_cudaobjdetect opencv_cudaoptflow opencv_cudastereo opencv_cudawarping''')



includes = ['include', '/usr/local/include/opencv2', '/usr/local/include/opencv']

# #cxx_flags = "-Wall -g -fno-inline -fopenmp"
# #cxx_flags = "-Wall -g -fno-inline"
# cxx_flags = "-Wall -O3 -fopenmp -std=c++11"
# link_flags = "-fopenmp"

# env = Environment(CPPPATH=includes, CXXFLAGS=cxx_flags, LINKFLAGS=link_flags, 
# 		LIBS=boost_libs + opencv_libs + cuda_opencv_libs,
# 		LIBPATH=['/usr/local/lib'],
# 		LD_LIBRARY_PATH=['/usr/local/lib'])

# #env.Program('bin/eco_train_gpu', eco_train_gpu)
# env.Program('bin/test', ['src/test_main.cpp'] + eco_filter_src)


# create a cuda environment
# env = Environment()
# env['CXXFLAGS'] = -std=c++11
# env.Program

cuda_env = Environment(CC = '/usr/bin/nvcc', CPPPATH = includes)
cuda_env.Append(LIBS = opencv_libs)
cuda_env.Tool('nvcc',toolpath = ['/home/ecestudent/nvcc/'])
# cuda_env.Append(LIBPATH = ['/usr/local/cuda/lib64'])
cuda_env.Program('test', ['src/main.cu', 'src/gpu_filter.cu'])



