import torch.nn as nn
from dgl.nn.pytorch.conv import APPNPConv
import torch as th
from torch import nn
from dgl import function as fn

import math

from torch import Tensor
from torch.nn.parameter import Parameter, UninitializedParameter
from torch.nn import functional as F
from torch.nn import init
from torch.nn.modules.module import Module
from torch.nn.modules.lazy import LazyModuleMixin


class APPNP(nn.Module):
    def __init__(self,
                 g,
                 in_feats,
                 hiddens,
                 n_classes,
                 activation,
                 feat_drop,
                 edge_drop,
                 alpha,
                 k):
        super(APPNP, self).__init__()
        self.g = g
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(Linear(in_feats, hiddens[0]))
        # hidden layers
        for i in range(1, len(hiddens)):
            self.layers.append(Linear(hiddens[i - 1], hiddens[i]))
        # output layer
        self.layers.append(Linear(hiddens[-1], n_classes))
        self.activation = activation
        if feat_drop:
            self.feat_drop = nn.Dropout(feat_drop)
        else:
            self.feat_drop = lambda x: x
        self.propagate = APPNPConv(k, alpha, edge_drop)
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, features, model_dic=None):
        # prediction step
        h = features
        h = self.feat_drop(h)
        if model_dic is not None:
            h = self.activation(self.layers[0](input=h,weigh=model_dic['layers.0.weight'], bias=model_dic['layers.0.bias']))
        else:
            h = self.activation(self.layers[0](input=h,weigh=None,bias=None))
        i = 1
        for layer in self.layers[1:-1]:
            if model_dic is not None:
                h = self.activation(layer(input=h,weigh=model_dic['layers.'+str(i)+'.weight'], bias=model_dic['layers.'+str(i)+'.bias']))
            else:
                h = self.activation(layer(input=h,weigh=None,bias=None))
            i += 1
        if model_dic is not None:
            h = self.layers[-1](input=self.feat_drop(h),weigh=model_dic['layers.'+str(i)+'.weight'], bias=model_dic['layers.'+str(i)+'.bias'])
        else:
            h = self.layers[-1](input=self.feat_drop(h),weigh=None,bias=None)
        # propagation step
        h = self.propagate(self.g, h)
        return h

class Linear(Module):
    r"""Applies a linear transformation to the incoming data: :math:`y = xA^T + b`

    This module supports :ref:`TensorFloat32<tf32_on_ampere>`.

    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If set to ``False``, the layer will not learn an additive bias.
            Default: ``True``

    Shape:
        - Input: :math:`(N, *, H_{in})` where :math:`*` means any number of
          additional dimensions and :math:`H_{in} = \text{in\_features}`
        - Output: :math:`(N, *, H_{out})` where all but the last dimension
          are the same shape as the input and :math:`H_{out} = \text{out\_features}`.

    Attributes:
        weight: the learnable weights of the module of shape
            :math:`(\text{out\_features}, \text{in\_features})`. The values are
            initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`, where
            :math:`k = \frac{1}{\text{in\_features}}`
        bias:   the learnable bias of the module of shape :math:`(\text{out\_features})`.
                If :attr:`bias` is ``True``, the values are initialized from
                :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                :math:`k = \frac{1}{\text{in\_features}}`

    Examples::

        >>> m = nn.Linear(20, 30)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([128, 30])
    """
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(th.empty((out_features, in_features), **factory_kwargs))
        if bias:
            self.bias = Parameter(th.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, weigh, bias, input: Tensor) -> Tensor:
        if weigh is not None:
            return F.linear(input, weigh, bias)
        else:
            return F.linear(input, self.weight, self.bias)
        

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )


class NonDynamicallyQuantizableLinear(Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None) -> None:
        super().__init__(in_features, out_features, bias=bias,
                         device=device, dtype=dtype)


class Bilinear(Module):
    r"""Applies a bilinear transformation to the incoming data:
    :math:`y = x_1^T A x_2 + b`

    Args:
        in1_features: size of each first input sample
        in2_features: size of each second input sample
        out_features: size of each output sample
        bias: If set to False, the layer will not learn an additive bias.
            Default: ``True``

    Shape:
        - Input1: :math:`(N, *, H_{in1})` where :math:`H_{in1}=\text{in1\_features}` and
          :math:`*` means any number of additional dimensions. All but the last dimension
          of the inputs should be the same.
        - Input2: :math:`(N, *, H_{in2})` where :math:`H_{in2}=\text{in2\_features}`.
        - Output: :math:`(N, *, H_{out})` where :math:`H_{out}=\text{out\_features}`
          and all but the last dimension are the same shape as the input.

    Attributes:
        weight: the learnable weights of the module of shape
            :math:`(\text{out\_features}, \text{in1\_features}, \text{in2\_features})`.
            The values are initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`, where
            :math:`k = \frac{1}{\text{in1\_features}}`
        bias:   the learnable bias of the module of shape :math:`(\text{out\_features})`.
                If :attr:`bias` is ``True``, the values are initialized from
                :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`, where
                :math:`k = \frac{1}{\text{in1\_features}}`

    Examples::

        >>> m = nn.Bilinear(20, 30, 40)
        >>> input1 = torch.randn(128, 20)
        >>> input2 = torch.randn(128, 30)
        >>> output = m(input1, input2)
        >>> print(output.size())
        torch.Size([128, 40])
    """
    __constants__ = ['in1_features', 'in2_features', 'out_features']
    in1_features: int
    in2_features: int
    out_features: int
    weight: Tensor

    def __init__(self, in1_features: int, in2_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(Bilinear, self).__init__()
        self.in1_features = in1_features
        self.in2_features = in2_features
        self.out_features = out_features
        self.weight = Parameter(th.empty((out_features, in1_features, in2_features), **factory_kwargs))

        if bias:
            self.bias = Parameter(th.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        bound = 1 / math.sqrt(self.weight.size(1))
        init.uniform_(self.weight, -bound, bound)
        if self.bias is not None:
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input1: Tensor, input2: Tensor) -> Tensor:
        return F.bilinear(input1, input2, self.weight, self.bias)

    def extra_repr(self) -> str:
        return 'in1_features={}, in2_features={}, out_features={}, bias={}'.format(
            self.in1_features, self.in2_features, self.out_features, self.bias is not None
        )



class LazyLinear(LazyModuleMixin, Linear):
    r"""A :class:`torch.nn.Linear` module where `in_features` is inferred.

    In this module, the `weight` and `bias` are of :class:`torch.nn.UninitializedParameter`
    class. They will be initialized after the first call to ``forward`` is done and the
    module will become a regular :class:`torch.nn.Linear` module. The ``in_features`` argument
    of the :class:`Linear` is inferred from the ``input.shape[-1]``.

    Check the :class:`torch.nn.modules.lazy.LazyModuleMixin` for further documentation
    on lazy modules and their limitations.

    Args:
        out_features: size of each output sample
        bias: If set to ``False``, the layer will not learn an additive bias.
            Default: ``True``

    Attributes:
        weight: the learnable weights of the module of shape
            :math:`(\text{out\_features}, \text{in\_features})`. The values are
            initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`, where
            :math:`k = \frac{1}{\text{in\_features}}`
        bias:   the learnable bias of the module of shape :math:`(\text{out\_features})`.
                If :attr:`bias` is ``True``, the values are initialized from
                :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                :math:`k = \frac{1}{\text{in\_features}}`


    """

    cls_to_become = Linear  # type: ignore[assignment]
    weight: UninitializedParameter
    bias: UninitializedParameter  # type: ignore[assignment]

    def __init__(self, out_features: int, bias: bool = True,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        # bias is hardcoded to False to avoid creating tensor
        # that will soon be overwritten.
        super().__init__(0, 0, False)
        self.weight = UninitializedParameter(**factory_kwargs)
        self.out_features = out_features
        if bias:
            self.bias = UninitializedParameter(**factory_kwargs)

    def reset_parameters(self) -> None:
        if not self.has_uninitialized_params() and self.in_features != 0:
            super().reset_parameters()

    def initialize_parameters(self, input) -> None:  # type: ignore[override]
        if self.has_uninitialized_params():
            with th.no_grad():
                self.in_features = input.shape[-1]
                self.weight.materialize((self.out_features, self.in_features))
                if self.bias is not None:
                    self.bias.materialize((self.out_features,))
                self.reset_parameters()

