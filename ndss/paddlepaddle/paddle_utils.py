
import paddle

############################## 相关utils函数，如下 ##############################
############################ PaConvert 自动生成的代码 ###########################

def _Tensor_view(self, *args, **kwargs):
    if args:
        if len(args)==1 and isinstance(args[0], (tuple, list, str)):
            return paddle.view(self, args[0])
        else:
            return paddle.view(self, list(args))
    elif kwargs:
        return paddle.view(self, shape_or_dtype = list(kwargs.values())[0])

setattr(paddle.Tensor, 'view', _Tensor_view)

def device2int(device):
    if isinstance(device, str):
        device = device.replace('cuda', 'gpu')
        device = device.replace('gpu:', '')
    return int(device)

def device2str(type=None, index=None, *, device=None):
    type = device if device else type
    if isinstance(type, int):
        type = f'gpu:{type}'
    elif isinstance(type, str):
        if 'cuda' in type:
            type = type.replace('cuda', 'gpu')
        if 'cpu' in type:
            type = 'cpu'
        elif index is not None:
            type = f'{type}:{index}'
    elif isinstance(type, paddle.CPUPlace) or (type is None):
        type = 'cpu'
    elif isinstance(type, paddle.CUDAPlace):
        type = f'gpu:{type.get_device_id()}'

    return type
############################## 相关utils函数，如上 ##############################

