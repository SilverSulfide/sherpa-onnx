./build/bin/sherpa-onnx \
  --tokens=./sherpa-onnx-streaming-zipformer-en-2023-06-26/tokens.txt \
  --encoder=./sherpa-onnx-streaming-zipformer-en-2023-06-26/encoder-epoch-99-avg-1-chunk-16-left-128.onnx \
  --decoder=./sherpa-onnx-streaming-zipformer-en-2023-06-26/decoder-epoch-99-avg-1-chunk-16-left-128.onnx \
  --joiner=./sherpa-onnx-streaming-zipformer-en-2023-06-26/joiner-epoch-99-avg-1-chunk-16-left-128.onnx \
  ./sherpa-onnx-streaming-zipformer-en-2023-06-26/test_wavs/0.wav
