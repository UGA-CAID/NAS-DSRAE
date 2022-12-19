import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable


def _concat(xs):
  return torch.cat([x.view(-1) for x in xs])   # view(-1), make x into one column and
                                               # calculte the rows by itself
                                               # cat: categrate all the x into a matrix

def _clip(grads, max_norm):
    total_norm = 0
    for g in grads:
        param_norm = g.data.norm(2)     # sqrt(sum(pow(g,2)))
        total_norm += param_norm ** 2
    total_norm = total_norm ** 0.5     # = sqrt(sum(param_norm^2 in different grad))
                                       # norm for all the grads
    clip_coef = max_norm / (total_norm + 1e-6)  # change a little
    if clip_coef < 1:
        for g in grads:
            g.data.mul_(clip_coef)
    return clip_coef


class Architect(object):

  def __init__(self, model, args):
    self.network_weight_decay = args.wdecay
    self.network_clip = args.clip
    self.model = model
    self.optimizer = torch.optim.Adam(self.model.arch_parameters(), lr=args.arch_lr, weight_decay=args.arch_wdecay)

  def _compute_unrolled_model(self, hidden, input, target, eta):
    loss, hidden_next = self.model._loss(hidden, input, target)   # model_search
    theta = _concat(self.model.parameters()).data   # concat all the params
    grads = torch.autograd.grad(loss, self.model.parameters())
    clip_coef = _clip(grads, self.network_clip)
    dtheta = _concat(grads).data + self.network_weight_decay*theta  # change, concat the grads
                                                                    # then plus the weight, which decayed with ratio
    unrolled_model = self._construct_model_from_theta(theta.sub(eta, dtheta))
    return unrolled_model, clip_coef

  def step(self,
          hidden_train, input_train, target_train,
          hidden_valid, input_valid, target_valid,
          network_optimizer, unrolled):
    eta = network_optimizer.param_groups[0]['lr']  # eta control the learning rate
    self.optimizer.zero_grad()
    if unrolled:
        hidden = self._backward_step_unrolled(hidden_train, input_train, target_train, hidden_valid, input_valid, target_valid, eta)
    else:
        hidden = self._backward_step(hidden_valid, input_valid, target_valid)
    self.optimizer.step()
    return hidden, None

  def _backward_step(self, hidden, input, target):
    loss, hidden_next = self.model._loss(hidden, input, target)
    loss.backward()
    return hidden_next

  def _backward_step_unrolled(self,
          hidden_train, input_train, target_train,
          hidden_valid, input_valid, target_valid, eta):
    unrolled_model, clip_coef = self._compute_unrolled_model(hidden_train, input_train, target_train, eta)
    unrolled_loss, hidden_next = unrolled_model._loss(hidden_valid, input_valid, target_valid)

    unrolled_loss.backward()
    dalpha = [v.grad for v in unrolled_model.arch_parameters()]
    dtheta = [v.grad for v in unrolled_model.parameters()]
    _clip(dtheta, self.network_clip)
    vector = [dt.data for dt in dtheta]
    implicit_grads = self._hessian_vector_product(vector, hidden_train, input_train, target_train, r=1e-2)

    for g, ig in zip(dalpha, implicit_grads):
      g.data.sub_(eta * clip_coef, ig.data)

    for v, g in zip(self.model.arch_parameters(), dalpha):
      if v.grad is None:
        v.grad = Variable(g.data)
      else:
        v.grad.data.copy_(g.data)
    return hidden_next

  def _construct_model_from_theta(self, theta):
    model_new = self.model.new()   # model_search
    model_dict = self.model.state_dict()  # params for save

    params, offset = {}, 0
    for k, v in self.model.named_parameters():   # layer name ane the iterator
      v_length = np.prod(v.size())  # v_len = v.size[0] * v.size[1]...
      params[k] = theta[offset: offset+v_length].view(v.size()) # kth params transferred into v.size
      offset += v_length  # move flag

    assert offset == len(theta)  # check if the length is same as offset
    model_dict.update(params)
    model_new.load_state_dict(model_dict)  # load params
    return model_new.cuda()   # move the model to cuda

  def _hessian_vector_product(self, vector, hidden, input, target, r=1e-2):
    R = r / _concat(vector).norm()
    for p, v in zip(self.model.parameters(), vector):
      p.data.add_(R, v)
    loss, _ = self.model._loss(hidden, input, target)
    grads_p = torch.autograd.grad(loss, self.model.arch_parameters())

    for p, v in zip(self.model.parameters(), vector):
      p.data.sub_(2*R, v)
    loss, _ = self.model._loss(hidden, input, target)
    grads_n = torch.autograd.grad(loss, self.model.arch_parameters())

    for p, v in zip(self.model.parameters(), vector):
      p.data.add_(R, v)

    return [(x-y).div_(2*R) for x, y in zip(grads_p, grads_n)]

