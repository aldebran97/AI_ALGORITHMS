# 如果需要大内存训练

## 1. 如果是mac系统，cpu是arm架构：
```text
-Xmx4G -Djavacpp.platform=linux-arm64 -Dorg.bytedeco.javacpp.maxbytes=0G -Dorg.bytedeco.javacpp.maxphysicalbytes=0G
```

## 2. 如果是linux/windows，cpu是x86_64架构：
```text
-Xmx4G -Dorg.bytedeco.javacpp.maxbytes=0G -Dorg.bytedeco.javacpp.maxphysicalbytes=0G
```

## 3. 如果系统虚拟内存管理很优秀，不会让系统卡顿，且允许分配足够大的虚拟内存，，例如macos:
```text
-Xmx4G -Dorg.bytedeco.javacpp.maxbytes=[?]G -Dorg.bytedeco.javacpp.maxphysicalbytes=[?]G
注：?根据实际需要和系统允许的内存量决定
```

## 4.如果指定内存映射文件(充当内存)，可以用指令提前生成，也可以依靠JAVA自动生成
```java
activateMMapFile(File f, long size)
```
```text
可以使用指令提前生成(例如40GB)：
dd if=/dev/zero of=tmpFile bs=1G count=40
```