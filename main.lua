-- Copyright (c) 2015-present, Facebook, Inc.
-- All rights reserved.
--
-- This source code is licensed under the BSD-style license found in the
-- LICENSE file in the root directory of this source tree. An additional grant
-- of patent rights can be found in the PATENTS file in the same directory.

require('xlua')
require('paths')
local tds = require('tds')
paths.dofile('data.lua')

local function train(epoch, words)
  local N = math.ceil(words:size(1) / g_params.batchsize)
  local cost = 0
  local smooth_cost = 0
  local y = torch.ones(1)
  --local input = torch.CudaTensor(g_params.batchsize, g_params.edim)
  --local target = torch.CudaTensor(g_params.batchsize)
  --local context = torch.CudaTensor(g_params.batchsize, g_params.memsize)
  --local time = torch.CudaTensor(g_params.batchsize, g_params.memsize)
  local input = torch.DoubleTensor(g_params.batchsize, g_params.edim)
  local target = torch.DoubleTensor(g_params.batchsize)
  local context = torch.DoubleTensor(g_params.batchsize, g_params.memsize)
  local time = torch.DoubleTensor(g_params.batchsize, g_params.memsize)
  input:fill(g_params.init_hid)
  for t = 1, g_params.memsize do
    time:select(2, t):fill(t)
  end
  for n = 1, N do
    --if g_params.show then xlua.progress(n, N) end
    for b = 1, g_params.batchsize do
      local m = math.random(g_params.memsize + 1, words:size(1)-1)
      target[b] = words[m+1]
      context[b]:copy(words:narrow(1, m - g_params.memsize + 1, g_params.memsize))
    end
    local x = {input, target, context, time}
    local out = g_model:forward(x)
    cost = cost + out[1]
    if smooth_cost == 0 then smooth_cost = out[1]
    else smooth_cost = 0.99 * smooth_cost + 0.01 * out[1] end
    if g_params.show then 
      if n % g_params.display == 0 then
        print(string.format('ep: %d(%d/%d) lr: %.5f, loss: %04.4f, smooth-loss: %04.4f, smooth-pplx: %04.4f',
                            epoch, n, N, g_params.dt, out[1], smooth_cost, math.exp(smooth_cost/g_params.batchsize)))
        io.flush()
      end
    end
    g_paramdx:zero()
    g_model:backward(x, y)
    local gn = g_paramdx:norm()
    if gn > g_params.maxgradnorm then
      g_paramdx:mul(g_params.maxgradnorm / gn)
    end
    g_paramx:add(g_paramdx:mul(-g_params.dt))
  end
  return cost/N/g_params.batchsize
end

local function test(epoch, words)
  local N = math.ceil(words:size(1) / g_params.batchsize)
  local cost = 0
  --local input = torch.CudaTensor(g_params.batchsize, g_params.edim)
  --local target = torch.CudaTensor(g_params.batchsize)
  --local context = torch.CudaTensor(g_params.batchsize, g_params.memsize)
  --local time = torch.CudaTensor(g_params.batchsize, g_params.memsize)
  local input = torch.DoubleTensor(g_params.batchsize, g_params.edim)
  local target = torch.DoubleTensor(g_params.batchsize)
  local context = torch.DoubleTensor(g_params.batchsize, g_params.memsize)
  local time = torch.DoubleTensor(g_params.batchsize, g_params.memsize)
  input:fill(g_params.init_hid)
  for t = 1, g_params.memsize do
    time:select(2, t):fill(t)
  end
  local m = g_params.memsize + 1
  for n = 1, N do
    --if g_params.show then xlua.progress(n, N) end
    for b = 1, g_params.batchsize do
      target[b] = words[m+1]
      context[b]:copy(words:narrow(1, m - g_params.memsize + 1, g_params.memsize))
      m = m + 1
      if m > words:size(1)-1 then
        m = g_params.memsize + 1
      end
    end
    local x = {input, target, context, time}
    local out = g_model:forward(x)
    cost = cost + out[1]
  end
  return cost/N/g_params.batchsize
end

local function run()
  for ep = 1, g_params.epochs do
    local c, ct
    c = train(ep, g_words_train)
    ct = test(ep, g_words_valid)
    -- Logging
    local m = #g_log_cost+1
    g_log_cost[m] = {m, c, ct}
    g_log_perp[m] = {m, math.exp(c), math.exp(ct)}
    local stat = {perplexity = math.exp(c) , epoch = m,
        valid_perplexity = math.exp(ct), LR = g_params.dt}
    if g_params.test then
      local ctt = test(g_words_test)
      table.insert(g_log_cost[m], ctt)
      table.insert(g_log_perp[m], math.exp(ctt))
      stat['test_perplexity'] = math.exp(ctt)
    end
    print(string.format('ep: %d val-loss: %04.4f, val-pplx: %04.4f',
                         ep, ct, math.exp(ct)))
    --print(stat)

    -- Learning rate annealing
    if m > 1 and g_log_cost[m][3] > g_log_cost[m-1][3] * 0.9999 then
      g_params.dt = g_params.dt / 1.5
      if g_params.dt < 1e-5 then break end
    end
  end
end

local function save(path)
  local d = {}
  d.params = g_params
  d.paramx = g_paramx:float()
  d.log_cost = g_log_cost
  d.log_perp = g_log_perp
  torch.save(path, d)
end

--------------------------------------------------------------------
--------------------------------------------------------------------
-- model params:
local cmd = torch.CmdLine()
cmd:option('--gpu', -1, 'GPU id to use')
cmd:option('--data', '/storage/ptb/data/', 'path to data')
cmd:option('--edim', 150, 'internal state dimension')
cmd:option('--lindim', 75, 'linear part of the state')
cmd:option('--init_std', 0.05, 'weight initialization std')
cmd:option('--init_hid', 0.1, 'initial internal state value')
cmd:option('--sdt', 0.01, 'initial learning rate')
cmd:option('--maxgradnorm', 50, 'maximum gradient norm')
cmd:option('--memsize', 100, 'memory size')
cmd:option('--nhop', 6, 'number of hops')
cmd:option('--batchsize', 128)
cmd:option('--show', true, 'print progress')
cmd:option('--load', '', 'model file to load')
cmd:option('--save', '', 'path to save model')
cmd:option('--epochs', 100)
cmd:option('--test', false, 'enable testing')
cmd:option('--display', 10, 'display interval')
g_params = cmd:parse(arg or {})

print(g_params)
if g_params.gpu ~= -1 then
  cutorch.setDevice(g_params.gpu)
end

g_vocab = tds.hash()
g_ivocab = tds.hash()
g_ivocab[#g_vocab+1] = '<eos>'
g_vocab['<eos>'] = #g_vocab+1

g_words_train =g_read_words(string.format('%s/ptb.train.txt', g_params.data), g_vocab, g_ivocab)
g_words_valid =g_read_words(string.format('%s/ptb.valid.txt', g_params.data), g_vocab, g_ivocab)
g_words_test = g_read_words(string.format('%s/ptb.test.txt', g_params.data), g_vocab, g_ivocab)
g_params.nwords = #g_vocab
print('vocabulary size ' .. #g_vocab)

if g_params.gpu ~= -1 then require('cunn')
else require('nn') end
paths.dofile('model.lua')

g_model = g_build_model(g_params)
g_paramx, g_paramdx = g_model:getParameters()
g_paramx:normal(0, g_params.init_std)
if g_params.load ~= '' then
  local f = torch.load(g_params.load)
  g_paramx:copy(f.paramx)
end

g_log_cost = {}
g_log_perp = {}
g_params.dt = g_params.sdt

print('starting to run....')
run()

if g_params.save ~= '' then
  save(g_params.save)
end
