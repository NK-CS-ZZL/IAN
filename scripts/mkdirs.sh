if [ ! -d "./data/any2any/train/input" ];then
mkdir -p "./data/any2any/train/input"
fi
if [ ! -d "./data/any2any/train/target" ];then
mkdir -p "./data/any2any/train/target"
fi
if [ ! -d "./data/any2any/train/depth" ];then
mkdir -p "./data/any2any/train/depth"
fi
if [ ! -d "./data/any2any/train/normals" ];then
mkdir -p "./data/any2any/train/normals"
fi

if [ ! -d "./data/one2one/train/input" ];then
mkdir -p "./data/one2one/train/input"
fi
if [ ! -d "./data/one2one/train/target" ];then
mkdir -p "./data/one2one/train/target"
fi
if [ ! -d "./data/one2one/train/depth" ];then
mkdir -p "./data/one2one/train/depth"
fi
if [ ! -d "./data/one2one/train/normals" ];then
mkdir -p "./data/one2one/train/normals"
fi

if [ ! -d "./data/supplement/train/input" ];then
mkdir -p "./data/supplement/train/input"
fi
if [ ! -d "./data/supplement/train/target" ];then
mkdir -p "./data/supplement/train/target"
fi
if [ ! -d "./data/supplement/train/depth" ];then
mkdir -p "./data/supplement/train/depth"
fi
if [ ! -d "./data/supplement/train/normals" ];then
mkdir -p "./data/supplement/train/normals"
fi

if [ ! -d "./data/one2one/validation/input" ];then
mkdir -p "./data/one2one/validation/input"
fi
if [ ! -d "./data/one2one/validation/target" ];then
mkdir -p "./data/one2one/validation/target"
fi
if [ ! -d "./data/one2one/validation/depth" ];then
mkdir -p "./data/one2one/validation/depth"
fi
if [ ! -d "./data/one2one/validation/normals" ];then
mkdir -p "./data/one2one/validation/normals"
fi

if [ ! -d "./data/test/input" ];then
mkdir -p "./data/test/input"
fi
if [ ! -d "./data/test/depth" ];then
mkdir -p "./data/test/depth"
fi
if [ ! -d "./data/test/normals" ];then
mkdir -p "./data/test/normals"
fi
