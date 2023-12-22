import copy
import arithmetic_compressor.arithmetic_coding as AE
from arithmetic_compressor.util import *

# Compress using arithmetic encoding


class AECompressor_RNN:
  def __init__(self) -> None:
    pass

  def compress(self, data, probability):
    encoder = AE.Encoder()

    for symbol, prob in zip(data,probability):
      # Use the model to predict the probability of the next symbol
      #probability =  {  1: Range(0, 2048), 0: Range(2048, 4096)}
      #print("symbol:{} probability:{}".format(symbol,prob))
      # encode the symbol
      encoder.encode_symbol(prob, symbol)

    encoder.finish()
    return encoder.get_encoded()

  def decompress(self, encoded, probability):
    decoded = []
    decoder = AE.Decoder(encoded)
    for prob in probability:
      # probability of the next symbol

      # decode symbol
      symbol = decoder.decode_symbol(prob)

      decoded += [symbol]
    return decoded