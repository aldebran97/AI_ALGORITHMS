### 1. 创建文件作为内存映射文件，JAVA自动创建或者使用Linux指令预先创建

dd if=/dev/zero of=tmpFile bs=1G count=10

上述命令个生成10G的文件tmpFile

