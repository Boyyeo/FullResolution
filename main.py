
'''
from arithmetic_compressor import AECompressor
from arithmetic_compressor.models import\
   BinaryPPM,\
   BaseBinaryModel

# create the model
model = BaseBinaryModel(update_rate=6)

# create an arithmetic coder
coder = AECompressor(model)

# encode some data
data = [0,0,0,1,0,0,0,1,0,1,0,1,0,0,1,0,1,0,1]
compressed = coder.compress(data)
# print the compressed data
print(compressed) # => [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1]
print("original:{} compressed:{}".format(len(data),len(compressed)))

decompressed = coder.decompress(compressed,length_encoded=len(data))
print("decompressed message:",decompressed)
print("Equal:{}".format((data==decompressed)))

print("gg:",1 - 1 / (1 << 10))

'''

from arithmetic_compressor import AECompressor
from arithmetic_compressor.models import\
   BinaryPPM,\
   BaseBinaryModel, SimpleAdaptiveModel

# create the model
model = SimpleAdaptiveModel({'0': 0.5, '1': 0.5},update_rate=0.9825)

# create an arithmetic coder
coder = AECompressor(model)

# encode some data
data = '0001000101010010101'
compressed = coder.compress(data)
# print the compressed data
decompressed = coder.decompress(compressed,length_encoded=len(data))
decompressed_string = ''.join(decompressed)
compressed_string = ''.join([str(s) for s in compressed])
print("compressed message:",compressed)
print("decompressed message:",decompressed_string)
print("Equal:{}".format((data==decompressed_string)))
print("original:{} compressed:{}".format(len(data),len(compressed)))

