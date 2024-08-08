
# https://www.youtube.com/watch?v=EJRFP3WmS6Q
import math

# primer valor de la dupla es la suma calucilada (SUM weight* w_input)  y el target
input_data = [(0.26,1),
              (0.20,0),
              (0.48,1),
              (0.30,0)]

def cross_entropy (input_data):
    loss=0
    n= len(input_data)
    for entry in input_data:
        w_sum = entry[0] # w_sum
        y = entry[1] #target
        # formula de entropy loss
        loss += -(y * math.log10(w_sum) + (1-y) * math.log10(1-w_sum))
        print(-(y * math.log10(w_sum) + (1-y) * math.log10(1-w_sum)))
    return loss / n  # average o mean
error_term = cross_entropy(input_data)
print("loss error or cost = " + str(error_term))
