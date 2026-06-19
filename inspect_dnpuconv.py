import inspect

from brainspy.processors.modules.conv import DNPUConv2d

print("DNPUConv2d:", DNPUConv2d)

print("\nSignature:")
print(inspect.signature(DNPUConv2d.__init__))

print("\nSource __init__:")
print(inspect.getsource(DNPUConv2d.__init__))

print("\nSource forward:")
print(inspect.getsource(DNPUConv2d.forward))

print("\nSource preprocess:")
print(inspect.getsource(DNPUConv2d.preprocess))

print("\nSource postprocess:")
print(inspect.getsource(DNPUConv2d.postprocess))
