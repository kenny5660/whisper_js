# Whisper weights converter

Python script for converting Whisper model weights from Torch format (.pt) to TensorFlow format (.h5).

## Usage 

4 optional arguments:

**--model**           - название весов для скачивания, затем конвертации (по умолчанию tiny) (если указан --local_path данный аргумент игнорируется).  
**--local_path, -l**  - если веса расположены локально, указываем путь до них. Иначе скачиваем.  
**--compression, -c** - способ сжатия весов (по умолчанию "lzf").  
**--output_dir**      - путь куда сложить конвертированные веса. Если не указан, веса складываются в локальную директорию.  

```
usage: convert_weights.py [-h] [--model {tiny.en,tiny,base.en,base,small.en,small,medium.en,medium,large}] [--compression {lzf,gzip,None}] [--local_path LOCAL_PATH] [--output_dir OUTPUT_DIR]
```

## Example
```
python convert_weights.py --model medium
python convert_weights.py -l C:\Documents\tiny.en.pt
python convert_weights.py -l C:\Documents\tiny.en.pt -c None --output_dir C:\Documents\something\
```