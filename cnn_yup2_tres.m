function cnn_yup2_tres(varargin)


run(fullfile(fileparts(mfilename('fullpath')), ...
 'matconvnet','matlab', 'vl_setupnn.m')) ;

addpath('network_surgery');

addpath('..')

opts = cnn_setup_environment();
opts.train.gpus = [1  ];

model = 'imagenet-vgg-m-2048'; opts.layer = 14;
model = 'imagenet-vgg-verydeep-16';
model = 'imagenet-resnet-50-dag';
opts.model = fullfile(opts.modelPath, [model '.mat']) ;

opts.numTrain = 3;
opts.numTest = 27;
opts.randSplit = 1;

opts.dataDir = fullfile(opts.dataPath, 'YUP2') ;
opts.imageDir = fullfile(opts.dataDir, 'jpegs_256') ;

opts.nSplit = 3;
opts.dropOutRatio = 0.0;
injectDropout = 0 ;
opts.inputdim  = [ 224,  224, 3] ;
opts.train.epochFactor = 50;
opts.train.batchSize = 256  ;
opts.train.numSubBatches = ceil( 64  / max(1,numel(opts.train.gpus))) ;
opts.train.numEpochs = 5
nFrames = 1 ;
doMultiTask = 0;
convertPool2D3D = 0 ;
convertFilters2D3D = 0 ;

injectTemporalResnet = 0 ; usePretrained = 0 ; addPool3D = 2 ;

injectTemporalSumDiff = 0 ;
TdownSample = 0 ; 
Tstride = 5:15 ;

method='tres-sumdif';
initStride= 1;
initRes = 'noise';
opts.train.numValFrames = 16 + TdownSample * 2 * injectTemporalResnet / initStride;
fuseAt = {''};
model = [model method initRes num2str(injectTemporalResnet) '-initstride=' num2str(initStride) '-TdownSample=' num2str(TdownSample)  '-f25noCtr'  '-split=' num2str(opts.nSplit) ...
  '-fuseOnly=' num2str(opts.train.batchSize) ...
'-bs=' num2str(opts.train.batchSize) ...
'-sub=' num2str(opts.train.numSubBatches*max(numel(opts.train.gpus),1)), ...
'-addPool3D=' num2str(addPool3D) ...
       '-usePretrained=' num2str(usePretrained) ...
    '-nFrames=' num2str(nFrames), ...    
    '-dr' num2str(opts.dropOutRatio)];

opts.dataDir = fullfile(opts.dataDir, ['yup2-numTrain=' num2str(opts.numTrain) '-numTest=' num2str(opts.numTest) '-randSplit=' num2str(opts.randSplit)]) ;

opts.imdbPath = ['numTrain=' num2str(opts.numTrain) '-numTest=' num2str(opts.numTest) '-randSplit=' num2str(opts.randSplit)] ;
if ~exist(opts.imdbPath, 'dir'), mkdir(opts.imdbPath) ; end

if usePretrained
  if opts.nSplit==1
    opts.model = fullfile(opts.dataDir, ['imagenet-resnet-50-dagtres-0-f25noCtr-split=2-bs=256-sub=16-convertPool2D3D=0-convertFilters2D3D-noPad=0-usePretrained=0-nFrames=1-dr0/net-epoch-1.mat']) ;
  else
    opts.model = fullfile(opts.dataDir, ['imagenet-resnet-50-dagtres-sumdifnoise0-initstride=1-TdownSample=0-f25noCtr-split=' num2str(opts.nSplit) '-fuseOnly=256-bs=256-sub=64-addPool3D=0-usePretrained=0-nFrames=1-dr0/net-epoch-3.mat']);
  end
  
end

opts.imdbPath = fullfile(opts.imdbPath,'imdb.mat') ;

opts.expDir = fullfile(opts.dataDir, model);

if ~exist(opts.expDir, 'dir'), mkdir(opts.expDir) ; end


opts.train.learningRate = .01 * [1e-2*ones(1, 4) 1e-3*ones(1, 2)  1e-4*ones(1,2) ]  ;


if strfind(model, 'res')
  opts.train.learningRate = 10 * [ 1e-3*ones(1,1) 1e-4*ones(1, 1) 1e-5*ones(1,1)]  ;
%   opts.train.learningRate = logspace(-3, -5, 20) *10;
end


opts.train.augmentation = 'randCropFlipStretch';
opts.train.augmentation = 'multiScaleRegular';

opts.train.augmentation = 'f25';
if usePretrained 
  opts.train.augmentation = 'f25noCtr';
end

opts.train.plotDiagnostics = 0;
opts.train.continue = 1 ;
opts.train.prefetch = 1 ;
opts.train.expDir = opts.expDir ;

opts.train.numAugments = 1;

opts.train.frameSample = 'random';
opts.train.nFramesPerVid = 1;
opts.train.uniformAugments  = false;

[opts, varargin] = vl_argparse(opts, varargin) ;

% -------------------------------------------------------------------------
%                                                   Database initialization
% -------------------------------------------------------------------------

if exist(opts.imdbPath)
 tic; imdb = load(opts.imdbPath) ; toc
 imdb.imageDir = opts.imageDir;
else
  imdb = cnn_yup2_setup_data_imgs(fullfile(opts.dataPath, 'YUP2'), opts) ;
  save(opts.imdbPath, '-struct', 'imdb', '-v6') ;
end


net = load(opts.model);
if isfield(net, 'net'), net=net.net;end


if isstruct(net.layers)
  % replace 1000-way imagenet classifiers
  for p = 1 : numel(net.params)
    sz = size(net.params(p).value);
    if any(sz == 1000)
      sz(sz == 1000) = 20;
      fprintf('replace classifier layer of %s\n', net.params(p).name);
      if numel(sz) > 2
         net.params(p).value = 0.01 * randn(sz,  class(net.params(p).value));
      else
         net.params(p).value = zeros(sz,  class(net.params(p).value));
      end
    end
  end
  net.meta.normalization.border = [256 256] - net.meta.normalization.imageSize(1:2);
  net = dagnn.DagNN.loadobj(net);
  if strfind(model, 'bnorm')
    net = insert_bnorm_layers(net) ;
  end
else
  if isfield(net, 'meta'),
    netNorm = net.meta.normalization;
  else
    netNorm = net.normalization;
  end
  if(netNorm.imageSize(3) == 3)
    net.meta.normalization.averageImage = [];
    net.meta.normalization.border = [256 256] - netNorm.imageSize(1:2);
    net = replace_last_layer(net, [1 2], [1 2], 20, opts.dropOutRatio);
  end
  if strfind(model, 'bnorm')
    net = insert_bnorm_layers(net) ;
  end

  net = dagnn.DagNN.fromSimpleNN(net) ;

end


if injectTemporalResnet
  [ net ] = insert_temporal_res_layers( net,  'insertSOE', 0, 'numLayers', injectTemporalResnet , ...
    'TdownSample', TdownSample, 'initRes', initRes, 'initStride', initStride);
end
if injectTemporalSumDiff
    [ net ] = insert_temporal_sumdiff_layers( net, 'nLayers' , injectTemporalSumDiff );
end

net = dagnn.DagNN.setLrWd(net);

if convertPool2D3D
  poolCtr = 0;
  pool_layers = find(arrayfun(@(x) isa(x.block,'dagnn.Pooling') , net.layers)) ;
  pool_layers = pool_layers(end-convertPool2D3D+1:end);
  if isempty(strfind(opts.model, 'res'))
    for l=pool_layers
      if isa(net.layers(l).block, 'dagnn.Pooling')

  if convertFilters2D3D
      net.layers(l-2).params
      block = dagnn.Conv3D() ;   block.net = net ;
      kernel = net.params(net.getParamIndex(net.layers(l-2).params{1})).value;
      bfilter = ones(3,1);
      bfilter = bfilter/sum(bfilter);
      bfilter = permute(bfilter, [5 2 3 4 1]);

      sz = size(kernel);
      kernel = bsxfun(@times, kernel, bfilter);
      kernel = permute(kernel, [1 2 5 3 4]);
      net.params(net.getParamIndex(net.layers(l-2).params{1})).value = kernel;
      pads = size(kernel); pads = ceil(pads(1:3) / 2) - 1
      block.pad = [pads(1),pads(1), pads(2),pads(2), 0,0]; % 3D lower/higher padding
      block.stride = [1 1 1];          % 3D stride 
      net.layers(l-2).block = block;
  end
      if strfind(net.layers(l).name,'pool5'),
        poolSz = ceil(nFrames - (2* convertPool2D3D ));
      else
        continue;
      end
        block = dagnn.Pooling3D() ;   block.net = net ;
        block.method = 'max' ;
        block.poolSize = [net.layers(l).block.poolSize, poolSz];          % 3D pooling window size
        block.pad = [net.layers(l).block.pad, 0,0]; % 3D lower/higher padding
        block.stride = [net.layers(l).block.stride, 2];          % 3D stride 
        net.layers(l).block = block;
      end
    end
  else
    conv_layers = find(arrayfun(@(x) isa(x.block,'dagnn.Conv') && isequal(x.block.size(1),3) ... 
      &&  x.block.hasBias && isequal(x.block.stride(1),1) , net.layers)) ;
    for l=conv_layers
       disp(['converting' net.layers(l).name ' to ConvTime'])
       block = dagnn.Conv3D() ;   block.net = net ;
       kernel = net.params(net.getParamIndex(net.layers(l).params{1})).value;
       sz = size(kernel); 
       kernel = cat(5, kernel/3, kernel/3, kernel/3  );
       kernel = permute(kernel, [1 2 5 3 4]);     
       net.params(net.getParamIndex(net.layers(l).params{1})).value = kernel;
       pads = size(kernel); pads = ceil(pads(1:3) / 2) - 1;   
       block.pad = [pads(1),pads(1), pads(2),pads(2) pads(3),pads(3)] ; 
       block.size = size(kernel);
       block.hasBias = net.layers(l).block.hasBias;
       if block.hasBias, 
          net.params(net.getParamIndex(net.layers(l).params{2})).value = ...
              net.params(net.getParamIndex(net.layers(l).params{2})).value' ; 
       end
       block.stride = [1 1 1]
       net.layers(l).block = block;    
    end
  end
end

if addPool3D

  poolLayers = {'res2a', 'res3a',  'res4a', 'res5a'; };
    poolLayers = {'pool5'; };

  for j=1:numel(poolLayers)

      i_pool = find(strcmp({net.layers.name},[poolLayers{j}  ]));         
      block = dagnn.PoolTime() ;
      block.poolSize = [1 Inf];  
      block.pad = [0 0 0 0]; 
      block.stride = [1 1];
      if addPool3D > 1
        block.method = 'max';     
      else
        block.method = 'avg';
      end
      name = [poolLayers{j} '_pool_time' ];
      disp(['injecting ' name ' as PoolTime'])
      net.addLayer(name, block, ...
                    [net.layers(i_pool).outputs], {name}) ; 

      % chain input of l that has layer as input
      for l = 1:numel(net.layers)    
          if ~strcmp(net.layers(l).name, name)
            sel = find(strcmp(net.layers(l).inputs, net.layers(i_pool).outputs{1})) ;
            if any(sel)
%               net.setLayerInputs( net.layers(l).name, {name}); 
                net.layers(l).inputs{sel} = name;
            end;   
          end
      end

  end
end % add pool3d


if injectTemporalResnet || convertPool2D3D || injectTemporalSumDiff || addPool3D
  opts.train.frameSample = 'temporalStrideRandom';
  opts.train.nFramesPerVid = nFrames;
  opts.train.temporalStride = Tstride;
  opts.train.valmode = 'temporalStrideRandom';
  opts.train.saveAllPredScores = 0;
  opts.train.denseEval = 1;
end 

if injectDropout

  pool5_layer = find(arrayfun(@(x) isa(x.block,'dagnn.Pooling'), net.layers)) ;
  conv_layers = pool5_layer(end);
  for i=conv_layers
    block = dagnn.DropOut() ;   block.rate = opts.dropOutRatio ;
    newName = ['drop_' net.layers(i).name];

    net.addLayer(newName, ...
      block, ...
      net.layers(i).outputs, ...
      {newName}) ;

      % Replace oldName with newName in all the layers
      for l = 1:numel(net.layers)-1
        for f = net.layers(i).outputs
           sel = find(strcmp(f, net.layers(l).inputs )) ;
           if ~isempty(sel)
            [net.layers(l).inputs{sel}] = deal(newName) ;
           end
        end
      end
  end
end

if doMultiTask
    numDatasets = 2;
    net.addVar('inputSet');
    net.removeLayer( net.layers(end).name) %remove prediction
    net.removeLayer( net.layers(end).name) % remove loss
    outputs = {}; labels = {};
    for k=1:numDatasets, 
        outputs{k} = sprintf('predIn_%d',k); 
        labels{k} = sprintf('label_%d',k);
    end
    tmp = [outputs; labels];
    net.addLayer('sliceMultitask', dagnn.SliceBatch(), ...
             [net.layers(end).outputs ,{'inputSet'}, {'label'}],  tmp(:)') ; 
    for k = 1:numDatasets
        input = outputs{k} ;
        output = sprintf('predOut_%d',k) ;
        name = sprintf('pred_layer_%d',k);
        in = size(net.params(net.getParamIndex('fc7f')).value,4)
        out = numel(imdb.classes.name{k})
        params(1).value = randn(1, 1, in, out, 'single')*0.01;
        params(2).value = zeros(1, out ,'single') ;
        params(1).name = [name 'f'];
        params(2).name = [name 'b'];
        net.addLayer(name,  dagnn.Conv(), {input}, {output}, {params.name}) ;            
        net.params(net.getParamIndex(params(1).name)).value = params(1).value ;
        net.params(net.getParamIndex(params(2).name)).value = params(2).value ;
        
        net.addLayer(sprintf('loss_%d',k), dagnn.Loss( 'loss', 'softmaxlog'), ...
             {output labels{k}}, sprintf('objective_%d',k)) ; 
    end
end
           

net.renameVar(net.vars(1).name, 'input');
for l = 1:numel(net.layers)
  if isa(net.layers(l).block, 'dagnn.DropOut')
    net.layers(l).block.rate = opts.dropOutRatio;
  end
end
net.layers(~cellfun('isempty', strfind({net.layers(:).name}, 'err'))) = [] ;

opts.train.derOutputs = {} ;
for l=numel(net.layers):-1:1
  if isa(net.layers(l).block, 'dagnn.Loss') && isempty(strfind(net.layers(l).name, 'err'))
      opts.train.derOutputs = {opts.train.derOutputs{:}, net.layers(l).outputs{:}, 1} ;
  end
  if isa(net.layers(l).block, 'dagnn.SoftMax') 
    net.removeLayer(net.layers(l).name)
    l = l - 1;
  end
end

if isempty(opts.train.derOutputs)
  net = dagnn.DagNN.insertLossLayers(net, 'numClasses', 20) ;
  fprintf('setting derivative for layer %s \n', net.layers(end).name);
  opts.train.derOutputs = {opts.train.derOutputs{:}, net.layers(end).outputs{:}, 1} ;
end

lossLayers = find(arrayfun(@(x) isa(x.block,'dagnn.Loss') && strcmp(x.block.loss,'softmaxlog'),net.layers));


net.addLayer('top1error', ...
             dagnn.Loss('loss', 'classerror'), ...
             net.layers(lossLayers(end)).inputs, ...
             'top1error') ;

net.addLayer('top5error', ...
             dagnn.Loss('loss', 'topkerror', 'opts', {'topK', 5}), ...
             net.layers(lossLayers(end)).inputs, ...
             'top5error') ;
           

net.print() ;   
net.rebuild() ;
if isempty(net.meta.normalization.averageImage)
  % compute image statistics (mean, RGB covariances etc)
  imageStatsPath = fullfile(opts.expDir, 'imageStats.mat') ;
  if exist(imageStatsPath)
    load(imageStatsPath, 'averageImage', 'rgbMean', 'rgbCovariance') ;
  else
    net.meta.normalization.averageImage = []; 
    net.meta.normalization.rgbVariance = [];
    tmp = opts.train.nFramesPerVid;
    opts.train.nFramesPerVid = 1;
    fn = getBatchWrapper_ucf101_imgs(net.meta.normalization, opts.numFetchThreads, opts.train) ;

    [averageImage, rgbMean, rgbCovariance] = getImageStats(imdb, fn);
      if ~exist(opts.expDir, 'dir'), mkdir(opts.expDir) ; end
    save(imageStatsPath, 'averageImage', 'rgbMean', 'rgbCovariance') ;
    opts.train.nFramesPerVid = tmp;
  end




  net.meta.normalization.averageImage = gather(mean(mean(averageImage,1),2)); 
end

net.meta.normalization.rgbVariance = [];

switch opts.nSplit
  case 1
    opts.train.train = find(ismember(imdb.images.set, [1 ])) ;
    opts.train.val = find(ismember(imdb.images.set, [2])) ;
  case 2
    opts.train.train = find(ismember(imdb.images.set, [3 ])) ;
    opts.train.val = find(ismember(imdb.images.set, [4])) ;
  case 3
    opts.train.train = find(ismember(imdb.images.set, [1 3])) ;
    opts.train.val = find(ismember(imdb.images.set, [2 4])) ;
end

opts.train.train = repmat(opts.train.train,1,opts.train.epochFactor);
net.meta.normalization.averageImage = mean(mean(net.meta.normalization.averageImage,1),2) ;

% opts.train.backpropDepth = 'pool5';
% opts.train.train = NaN;
% opts.train.valmode = 'centreSamplesFast'
% opts.train.valmode = '250samples'
opts.train.denseEval = 1;
net.conserveMemory = 1 ;
fn = getBatchWrapper_imgs(net.meta.normalization, opts.numFetchThreads, opts.train) ;
[info] = cnn_train_dag(net, imdb, fn, opts.train) ;

end

