# Copyright (C) 2021 Intel Labs
#
# BSD-3-Clause License
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
# 3. Neither the name of the copyright holder nor the names of its contributors
#    may be used to endorse or promote products derived from this software
#    without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
# THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS
# BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY,
# OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT
# OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
# OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
# EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Functions related to BNN to DNN model conversion.
#
# @authors: Julián Arias
#
# ===============================================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch


# --------------------------------------------------------------------------------
# Parameters used to define BNN layyers.
#    bnn_prior_parameters = {
#       "prior_mu": 0.0,
#       "prior_sigma": 1.0,
#       "posterior_mu_init": 0.0,
#       "posterior_rho_init": -4.0,
#       "type": "Reparameterization",  # Flipout or Reparameterization
# }


def dnn_linear_layer(d):
    if "Reparameterization" in d.__class__.__name__:
        type = "Reparameterization"
    #elif "Flipout" in d.__class__.__name__:
    #    type = "Flipout"
    else:
        raise('Unknown bayesian conversion strategy')

    layer_type = d.__class__.__name__.replace(type, '')
    layer_fn = getattr(torch.nn, layer_type)  # Get DNN layer

    dnn_layer = layer_fn(
        in_features=d.in_features,
        out_features=d.out_features,
        bias=d.bias,
    )
    sigma_weight = torch.log1p(torch.exp(d.rho_weight))
    weight = d.mu_weight + (sigma_weight * d.eps_weight.data.normal_())
    dnn_layer.weight = torch.nn.Parameter(weight)
    
    if d.bias:
        sigma_bias = torch.log1p(torch.exp(d.rho_bias))
        bias = d.mu_bias + (sigma_bias * d.eps_bias.data.normal_())
        dnn_layer.bias = torch.nn.Parameter(bias)

    
    
    
    return dnn_layer


def dnn_conv_layer(d):

    if "Reparameterization" in d.__class__.__name__:
        type = "Reparameterization"
    #elif "Flipout" in d.__class__.__name__:
    #    type = "Flipout"
    else:
        raise('Unknown bayesian conversion strategy')

    layer_type = d.__class__.__name__.replace(type, '')
    layer_fn = getattr(torch.nn, layer_type)  # Get DNN layer

    dnn_layer = layer_fn(
        in_channels=d.in_channels,
        out_channels=d.out_channels,
        kernel_size=d.kernel_size,
        stride=d.stride,
        padding=d.padding,
        dilation=d.dilation,
        groups=d.groups,
        bias=d.bias,
    )

    #Sampling layer distribution
    sigma_weight = torch.log1p(torch.exp(d.rho_kernel))
    eps_kernel = d.eps_kernel.data.normal_()
    weight = d.mu_kernel + (sigma_weight * eps_kernel)
    dnn_layer.weight = torch.nn.Parameter(weight)
    
    if d.bias:
        sigma_bias = torch.log1p(torch.exp(d.rho_bias))
        eps_bias = d.eps_bias.data.normal_()
        bias = d.mu_bias + (sigma_bias * eps_bias)
        dnn_layer.bias = torch.nn.Parameter(bias)

    return dnn_layer


def dnn_lstm_layer(d):
    if "Reparameterization" in d.__class__.__name__:
        type = "Reparameterization"
    #elif "Flipout" in d.__class__.__name__:
    #    type = "Flipout"
    else:
        raise('Unknown bayesian conversion strategy')

    layer_type = d.__class__.__name__.replace(type, '')
    layer_fn = getattr(torch.nn, layer_type)  # Get DNN layer
    dnn_layer = layer_fn(
        in_features=d.input_size,
        out_features=d.hidden_size,
        bias=d.bias,
    )
    #Sampling Linear layer corresponding to input-hidden weights
    sigma_weight = torch.log1p(torch.exp(d.ih.rho_weight))
    weight = d.ih.mu_weight + (sigma_weight * d.ih.eps_weight.data.normal_())
    dnn_layer.weight_ih_l0 = torch.nn.Parameter(weight)

    if d.bias:
        sigma_bias = torch.log1p(torch.exp(d.ih.rho_bias))
        bias = d.ih.mu_bias + (sigma_bias * d.ih.eps_bias.data.normal_())
        dnn_layer.bias_ih_l0 = torch.nn.Parameter(bias)
    
    #Sampling Linear layer corresponding to hidden-hidden weights
    sigma_weight = torch.log1p(torch.exp(d.hh.rho_weight))
    weight = d.hh.mu_weight + (sigma_weight * d.hh.eps_weight.data.normal_())
    dnn_layer.weight_hh_l0 = torch.nn.Parameter(weight)

    if d.bias:
        sigma_bias = torch.log1p(torch.exp(d.hh.rho_bias))
        bias = d.hh.mu_bias + (sigma_bias * d.hh.eps_bias.data.normal_())
        dnn_layer.bias_hh_l0 = torch.nn.Parameter(bias)
    
    return dnn_layer


# replaces linear and conv layers
def bnn_to_dnn(m):
    for name, value in list(m._modules.items()):
        if m._modules[name]._modules:
            bnn_to_dnn(m._modules[name])
        elif "Conv" in m._modules[name].__class__.__name__:
            setattr(
                m,
                name,
                dnn_conv_layer(
                    m._modules[name]))
        elif "Linear" in m._modules[name].__class__.__name__:
            setattr(
                m,
                name,
                dnn_linear_layer(
                    m._modules[name]))
        elif "LSTM" in m._modules[name].__class__.__name__:
            setattr(
                m,
                name,
                dnn_lstm_layer(
                    m._modules[name]))
        else:
            pass
    return