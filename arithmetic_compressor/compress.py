import copy
import arithmetic_compressor.arithmetic_coding as AE
from arithmetic_compressor.util import *

# Compress using arithmetic encoding


class AECompressor:
  def __init__(self, model, adapt=True) -> None:
    self.adapt = adapt
    # clone model, so we dont mutate original
    self.model = copy.deepcopy(model)
    self.__model = copy.deepcopy(model)  # for decoding

  def compress(self, data):
    encoder = AE.Encoder()

    for symbol in data:
      # Use the model to predict the probability of the next symbol
      probability = self.model.cdf()
      #probability =  {  1: Range(0, 2048), 0: Range(2048, 4096)}
      print("symbol:{} probability:{}".format(symbol,probability))
      # encode the symbol
      encoder.encode_symbol(probability, symbol)

      if self.adapt:
        # update the model with the new symbol
        self.model.update(symbol)
    encoder.finish()
    return encoder.get_encoded()

  def decompress(self, encoded, length_encoded):
    decoded = []
    model = self.__model
    decoder = AE.Decoder(encoded)
    for _ in range(length_encoded):
      # probability of the next symbol
      probability = model.cdf()

      # decode symbol
      symbol = decoder.decode_symbol(probability)

      if self.adapt:
        # update model
        model.update(symbol)

      decoded += [symbol]
    return decoded
