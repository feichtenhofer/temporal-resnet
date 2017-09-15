function [ net ] = insert_temporal_res_layers( net, varargin )


opts.resDepth = [2; 2; 3*ones(3,1)];
opts.resDepth = [1; 1; 1];
opts.insertSOE = false;
opts.TdownSample= 0;
opts.cudnnWorkspaceLimit = 1024*1024*1024 ; % 1GB
opts.numLayers = 4;
opts.learningRateTemporal = [1 2] ;
opts.injectSum = true;
opts.initRes = 'noise';
opts.initStride = 1;

opts = vl_argparse(opts, varargin) ;
if ~isa(net,'dagnn.DagNN')
  net = dagnn.DagNN.fromSimpleNN(net) ;
end


% i_conv = find(arrayfun(@(x) isa(x.block,'dagnn.Conv')  && ~isempty(strfind(x.name,'res')), net.layers)) ;
%   i_conv = find(arrayfun(@(x) isa(x.block,'dagnn.Conv') && isequal(x.block.size(1),3) ... 
%       && ~isempty(strfind(x.name,'branch2b'))  , net.layers)) ;
     i_conv = find(arrayfun(@(x) isa(x.block,'dagnn.Conv') && isequal(x.block.size(1),3) ... 
      && (~isempty(strfind(x.name,'a_branch2b')) | ~isempty(strfind(x.name,'c_branch2b'))), net.layers)) ;

%   i_conv=1  
% i_conv = i_conv(end-opts.numLayers+1:end);
%  i_conv = i_conv(vl_colsubset(1:numel(i_conv), opts.numLayers, 'Uniform'));
function ConvTime(name, ksize, appendReLU, downsample, bias, appendBnorm, init)
    if downsample
      pad = [0 0] ;
    else
      pad = [0 (ksize-1)/2] ;
    end
    if bias
      pars = {[name '_f'], [name '_b']} ;
    else
      pars = {[name '_f']} ;
    end
    
    doScale = 1;
    if appendBnorm
      net.addLayer([name '_bn'], ...
        dagnn.BatchNorm('numChannels', prevDepth), ...
        inputVar, ...
        [name '_bn'], ...
        {[name '_bn_w'], [name '_bn_b'], [name '_bn_m']}) ;
      p = net.getParamIndex(net.layers(end).params) ;
      params = net.layers(end).block.initParams() ;
      [net.params(p).value] = deal(params{:}) ;
      [net.params(p).learningRate] = deal(opts.learningRateTemporal(1), opts.learningRateTemporal(2), 0.3);
      [net.params(p).weightDecay] = deal(0, 0, 0);
      inputVar = [name '_bn'] ;
    end

    if appendReLU
      net.addLayer([name '_relu'] , ...
                   dagnn.ReLU(), ...
                   inputVar, ...
                   [name '_relu']) ;
      inputVar = [name '_relu'] ;
    end
    
    net.addLayer([name  '_conv'], ...
      dagnn.ConvTime('size', [1 ksize prevDepth outDepth], ...
      'stride', stride, ....
      'pad', pad, ...
      'hasBias', bias, ...
      'opts', {'cudnnworkspacelimit', opts.cudnnWorkspaceLimit}), ...
      inputVar, ...
      [name '_conv'], ...
      pars) ;
    p = net.getParamIndex(net.layers(end).params) ;
%     params = net.layers(end).block.initParams(.001) ;
    switch init
      case 1
        params = net.layers(end).block.initParamsDiffTime(.01);
      case 0
        params = net.layers(end).block.initParamsAvgTime(.01);
      case -1
        params = net.layers(end).block.initParams(.01) ;
    end
    [net.params(p).value] = deal(params{:}) ;
    if bias
      [net.params(p).learningRate] = deal(opts.learningRateTemporal(1), opts.learningRateTemporal(2));
    else
      [net.params(p).learningRate] = deal(opts.learningRateTemporal(1));
    end
    inputVar = [name '_conv'] ;
    prevDepth = depth ;
    
    inputVar = net.layers(end).outputs;

    if doScale
      
       net.addLayer([name  '_scale'], ...
           dagnn.Scale('size', [1 1 outDepth]), ...
                      inputVar, ...
                      [name '_scale'], ...
                      {[name '_scalef'], [name '_scaleb']}) ;
        p = net.getParamIndex(net.layers(end).params) ;
        params = net.layers(end).block.initParams(.01) ;
        [net.params(p).value] = deal(params{:}) ;
    end
    inputVar = net.layers(end).outputs;
end

function SOEBlock(name, bias, ksize, doPool, doScale, stride, outDepth)

    if bias
      pars = {[name '_f'], [name '_b']} ;
    else
      pars = {[name '_f']} ;
    end


    G3 = []; %G3Init3D(1) ;
    G2 =  G3Init3D(1) ;
    kernel = permute(cat(1,G3,G2), [3 2 4 1]);
    kernel = repmat(kernel, 1,1,3); %for rgb just replicate filter
   
    
    net.addLayer([name  '_sep1'], ...
      dagnn.Conv('size', size(kernel), ...
      'stride', [1 stride(2)], ....
      'pad', [0 0 (ksize-1)/2 (ksize-1)/2], ...
      'hasBias', 0), ...
      inputVar, ...
      [name '_sep1'], ...
      {[name '_sep1_f']}) ;
    p = net.getParamIndex(net.layers(end).params) ;
    [net.params(p).value] = kernel ;
    
    G3 = []; %G3Init3D(2)' ;
    G2 =  G3Init3D(2)' ;
    kernel = permute(cat(2,G3,G2), [1 3 4 2]);
    inputVar = net.layers(end).outputs;
    
    net.addLayer([name  '_sep2'], ...
      dagnn.Conv('size', size(kernel), ...
      'stride', [stride(1) 1], ....
      'pad', [(ksize-1)/2 (ksize-1)/2 0 0 ], ...
      'hasBias', 0), ...
      inputVar, ...
      {[name '_sep2_f']}, ...
      {[name '_sep2_f']}) ;
    p = net.getParamIndex(net.layers(end).params) ;
    [net.params(p).value] = kernel ;
    inputVar = net.layers(end).outputs;
    

    G3 = []; %G3Init3D(3) ;
    G2 =  G3Init3D(3) ;
    kernel = permute(cat(1,G3,G2), [3 2 4 1]);    
    net.addLayer([name  '_sep3'], ...
      dagnn.ConvTime('size', size(kernel), ...
      'stride', 1, ....
      'pad', [0 (ksize-1)/2], ...
      'hasBias', 0), ...
      inputVar, ...
      {[name '_sep3_f']}, ...
      {[name '_sep3_f']}) ;
    p = net.getParamIndex(net.layers(end).params) ;
    [net.params(p).value] = kernel ;
    inputVar = net.layers(end).outputs;

    
    pool_layers = find(arrayfun(@(x) isa(x.block,'dagnn.Pooling') , net.layers)) ;
    block = net.layers(pool_layers(1)).block; block.method = 'avg' ;
    net.addLayer( 'agg_SOE', block, ...
              inputVar,  ...
                'agg_SOE') ; 
    inputVar = net.layers(end).outputs;

    % steer block conv
    SOE_speeds = [1 3];
    SOE_orients = 6 ;
    kernel = init3DG3steerSet(SOE_speeds, SOE_orients, 'G3');
    kernel = permute(kernel, [3 4 1 2]);  
    net.addLayer([name  '_steer'], ...
      dagnn.Conv('size', size(kernel), ...
      'stride', 1, ....
      'hasBias', 0), ...
      inputVar, ...
      {[name '_steer']}, ...
      {[name '_steer_f']}) ;
    
    p = net.getParamIndex(net.layers(end).params) ;
    [net.params(p).value] = kernel ;
    inputVar = net.layers(end).outputs;
    thisDepth = size(kernel,4)

%     % marginalize block conv
%     SteerN = 4;
%     block = dagnn.Steer3D() ; block.N = SteerN;
%     block.addStaticAndEps = 0; block.epsilon =  [] ; %1e5 %
%     block.addStatic = 0;  block.subStatic = 0;
%     block.ksize_used_no_pad = (ksize-1)/2;
%     net.addLayer( [name  'Steer3D'] , block, ...
%               inputVar,  ...
%                 [name  'Steer3D']) ; 
% 
%           
%     inputVar = net.layers(end).outputs;
% 
%     thisDepth = size(kernel,4)/(SteerN) - 1
    net.addLayer([name '_bn'], ...
        dagnn.BatchNorm('numChannels', thisDepth), ...
        inputVar, ...
        [name '_bn'], ...
        {[name '_bn_w'], [name '_bn_b'], [name '_bn_m']}) ;
      p = net.getParamIndex(net.layers(end).params) ;
      params = net.layers(end).block.initParams() ;
      [net.params(p).value] = deal(params{:}) ;
      [net.params(p).learningRate] = deal(1, 2, 0.3);
      [net.params(p).weightDecay] = deal(0, 0, 0);
      inputVar = [name '_bn'] ;
     net.addLayer([name  '_conv'], ...
               dagnn.Conv('size', [1 1 thisDepth outDepth], ...
                          'stride', 1, ....
                          'pad', 0, ...
                          'hasBias', 1), ...
                          inputVar, ...
                          [name '_conv'], ...
                          {[name '_f'], [name '_b']}) ;
      p = net.getParamIndex(net.layers(end).params) ;
      params = net.layers(end).block.initParams(.001) ;
      [net.params(p).value] = deal(params{:}) ;
      
    pool1 = find(strcmp({net.layers.name},'pool1'));

    net.addLayer([name '_relu'] , ...
                 dagnn.ReLU(), ...
                 [name '_conv'], ...
                 [name '_relu']) ;
    inputVar = [name '_relu'] ;
    inputVar = net.layers(end).outputs;

    if doScale
      
       net.addLayer([name  '_scale'], ...
           dagnn.Scale('size', [1 1 outDepth]), ...
                      inputVar, ...
                      [name '_scale'], ...
                      {[name '_scalef'], [name '_scaleb']}) ;
        p = net.getParamIndex(net.layers(end).params) ;
        params = net.layers(end).block.initParams(.01) ;
        [net.params(p).value] = deal(params{:}) ;
    end
    inputVar = net.layers(end).outputs;

end
    
if 0    
    
    blk = dagnn.Conv() ;
    params(1).name = [name '_f'] ;
    params(2).name = [name '_1sep1_b'] ;
    net.addLayer('conv1_1sep1',  blk, ...
             [net.layers(1).inputs ], ...
            'conv1_1sep1', {params.name}) ;  
    G3 = [] %G3Init3D(1) ;
    G2 =  G3Init3D(1) ;
    kernel = permute(cat(1,G3,G2), [3 2 4 1]);
%     net.meta.normalization.imageSize(3) = 1;
    kernel = repmat(kernel, 1,1,3); %for rgb just replicate filter

    size(kernel)
    net.params(net.getParamIndex(params(1).name)).value = kernel ;
    net.layers(end).block.size=  size(kernel);
    pads = ceil(net.layers(end).block.size(1:2) / 2) - 1;
    
    net.meta.normalization.imageSize(1:2) =     net.meta.normalization.imageSize(1:2) + pads;
%     net.layers(end).block.pad = [pads(1),pads(1), pads(2),pads(2)]; % 2D lower/higher padding

    net.layers(end).block.pad = [0 0 0 0];
    net.layers(end).block.stride = [1 2];
    
    conv_layers = find(arrayfun(@(x) isa(x.block,'dagnn.Conv'),net.layers)) ;
%     net.removeLayer(net.layers(conv_layers(2)).name, false);
    
    blk = dagnn.Conv() ;
    params(1).name = 'conv1_1sep2_f' ;
    params(2).name = 'conv1_1sep2_b' ;
    net.addLayer( 'conv1_1sep2',  blk, ...
             [net.layers(end).outputs ], ...
            'conv1_1sep2', {params.name}) ;  
    G3 = [] %G3Init3D(2)' ;
    G2 =  G3Init3D(2)' ;
    kernel = permute(cat(2,G3,G2), [1 3 4 2]);
    size(kernel)
    net.layers(end).block.size = size(kernel);
    pads = ceil(net.layers(end).block.size(1:2) / 2) - 1;
    net.meta.normalization.imageSize(1:2) =     net.meta.normalization.imageSize(1:2) + pads;
    net.layers(1).block.pad = [0 0 0 0];

%     net.layers(end).block.pad = [pads(1),pads(1), pads(2),pads(2)]; % 2D lower/higher padding
    net.layers(end).block.pad = [0 0 0 0];
    net.layers(end).block.stride = [2 1];    
    net.params(net.getParamIndex(params(1).name)).value = kernel ;
%     net.params(net.getParamIndex(params(2).name)).value = zeros(1, net.layers(2).block.size(end) ,'single') ;  
    
    % temporal conv
    blk = dagnn.ConvTime() ;
    params(1).name = 'conv1_1sep3_f' ;
    params(2).name = 'conv1_1se3_b' ;
    net.addLayer( 'conv1_1sep3',  blk, ...
             [net.layers(end).outputs ], ...
            'conv1_1sep3', {params.name}) ;  
          
%     kernel = permute(G3Init3D(3),[1 2 3 5 4]);
    G3 = [] %G3Init3D(3) ;
    G2 =  G3Init3D(3) ;
    kernel = permute(cat(1,G3,G2), [3 2 4 1]);
    net.layers(end).block.size = size(kernel);
    pads = ceil(net.layers(end).block.size(1:2) / 2) - 1;
    net.layers(end).block.pad = [pads(1),pads(1), pads(2),pads(2)]; % 2D lower/higher padding
    net.layers(end).block.pad = [0 0 0 0];
    net.params(net.getParamIndex(params(1).name)).value = kernel ;
%     net.params(net.getParamIndex(params(2).name)).value = zeros(1, net.layers(3).block.size(end) ,'single') ;  
    
    % steer block conv
    blk = dagnn.Conv() ;
    params(1).name = 'conv1_1dt_f' ;
    params(2).name = 'conv1_1dt_b' ;
    net.addLayer('conv1_dt',  blk, ...
             [net.layers(end).outputs ], ...
            'conv1_dt', {params.name}) ;  
          

    
%     net.setLayerOutputs
%     kernel = single(repmat([-1 1],1,1,1,16));
%         kernel = single([-1 1]);
%     kernel = init3DG3steerSet((1:5)*.5, 12 , 'G2');
      SOE_speeds = [1 3];
      SOE_orients = 6 ;
        kernel = init3DG3steerSet(SOE_speeds, SOE_orients, 'G3');

%     kernel = init3DG3steerSet((1:8)*.5, 11);


%     kernel = kernel(:,5:end-8); % remove static and flicker chans
    kernel = permute(kernel, [3 4 1 2]);

    net.layers(end).block.size = size(kernel);
    pads = ceil(net.layers(end).block.size(1:2) / 2) - 1;
%     net.layers(4).block.pad = [pads(1),pads(1), pads(2),pads(2)]; % 2D lower/higher padding
    net.layers(end).block.pad = [0 0 0 0];
    net.params(net.getParamIndex(params(1).name)).value = kernel ;
    net.params(net.getParamIndex(params(2).name)).value = zeros(1, net.layers(end).block.size(end) ,'single') ;  
    
    % marginalize block conv
    SteerN = 4;
    block = dagnn.Steer3D() ; block.N = SteerN;
    block.addStaticAndEps = 0; block.epsilon =  [] ; %1e5 %
    block.addStatic = 0;  block.subStatic = 0;
    net.addLayer( 'Steer3D', block, ...
              'conv1_dt',  ...
                'Steer3D') ; 
      pool_layers = find(arrayfun(@(x) isa(x.block,'dagnn.Pooling') , net.layers)) ;
    block = net.layers(pool_layers(1)).block; block.method = 'avg' ;
    net.layers(pool_layers(1)).block.method = 'avg' ;
    net.addLayer( 'agg_SOE', block, ...
              'Steer3D',  ...
                'agg_SOE') ; 
          
    name = 'SOE';
    
%     net.addLayer([name '_difftime'] , ...
%     dagnn.DiffTime('doDiff',true, 'subSample', true), ...
%     net.layers(end).outputs , ...
%     [name '_difftime']) ;   
%     inputVar = {[name '_difftime']};
%     net.addLayer([name '_NormalizeLp'] , ...
%     dagnn.NormalizeLp('epsilon',1e5 , 'p', 1), ...
%     net.layers(end).outputs , ...
%     [name '_NormalizeLp']) ;   
    inputVar = net.layers(end).outputs;

    prevDepth = size(kernel,4)/(SteerN) - 1
    net.addLayer([name '_bn'], ...
        dagnn.BatchNorm('numChannels', prevDepth), ...
        inputVar, ...
        [name '_bn'], ...
        {[name '_bn_w'], [name '_bn_b'], [name '_bn_m']}) ;
      p = net.getParamIndex(net.layers(end).params) ;
      params = net.layers(end).block.initParams() ;
      [net.params(p).value] = deal(params{:}) ;
      [net.params(p).learningRate] = deal(1, 2, 0.3);
      [net.params(p).weightDecay] = deal(0, 0, 0);
      inputVar = [name '_bn'] ;
     net.addLayer([name  '_conv'], ...
               dagnn.Conv('size', [1 1 prevDepth 64], ...
                          'stride', 1, ....
                          'pad', net.layers(1).block.pad, ...
                          'hasBias', 1), ...
                          inputVar, ...
                          [name '_conv'], ...
                          {[name '_f'], [name '_b']}) ;
      p = net.getParamIndex(net.layers(end).params) ;
      params = net.layers(end).block.initParams(.0001) ;
      [net.params(p).value] = deal(params{:}) ;
      
    pool1 = find(strcmp({net.layers.name},'pool1'));

    net.addLayer([name '_relu'] , ...
                 dagnn.ReLU(), ...
                 [name '_conv'], ...
                 [name '_relu']) ;
    inputVar = [name '_relu'] ;
      
     res2 = find(arrayfun(@(x) isa(x.block,'dagnn.Conv') && isequal(x.block.size(1),1) ... 
      && ~isempty(strfind(x.name,'res2a_branch2a'))  , net.layers)) ;
%   

      sumVar = net.layers(res2).inputs ;
      net.addLayer([name '_sum'] , ...
        dagnn.Sum('pickTemporalCentre', true), ...
        {sumVar{:}, inputVar}, ...
        [name '_sum']) ;
      inputVar = [name '_sum'] ;
%       net.layers(res2).outputs = {inputVar} ;
      
        tmpInputVar = inputVar;
        % chain input of l that has layer as input
        for l = 1:numel(net.layers)    
          for input = net.layers(res2).outputIndexes
            sel = find(isequal(net.layers(l).inputIndexes, input)) ;
            if ~isempty(sel) && ~strcmp(net.layers(l).name, [name '_sum']);
              [net.layers(l).inputs{sel}] = deal(tmpInputVar) ;
              inputVar = net.layers(l).outputs{1} ;
            end
          end
        end
      
    opts.train.backpropDepth = 'Steer3D';

else
ctr = 0;
%   i = i_conv(1)   ;
  for i = i_conv
    ctr = ctr +1;
  %     if ~isa(net.layers(i).block,'dagnn.Conv'), continue; end

        for jj = i:-1:1
              p_from = net.layers(jj).params; 
              if numel(p_from) == 1 ||  numel(p_from) == 2, break; end;
        end

          depth = size(net.params(net.getParamIndex(p_from{1})).value,3);

%       depth = size(net.params( net.getParamIndex( net.layers(i_conv(i+1)).params{1})).value,3) ;
  %     sumDepth = size(net.params( net.getParamIndex( net.layers(i_conv(i+1)).params{1})).value,3) ;
  
      prevDepth = depth ;
      outDepth=  size(net.params(net.getParamIndex(p_from{1})).value,4);
%       inputVar = net.layers(i).outputs ;

      inputVar = net.layers(i).inputs ;
      stride = net.layers(i).block.stride;
      for l = 1 : opts.resDepth(1)
        if strcmp(opts.initRes, 'sumdif')
          init = mod(ctr,2);
        else
          init= -1;
        end
        name = sprintf('conv_time_%s', net.layers(i).name)  ;
        if opts.insertSOE
          SOEBlock(name, false, 7, 1, true, stride, outDepth)
        else
          ConvTime([name], 3, true, opts.TdownSample, true, 1, init) ;
        end
      end
  %     if s==1, i = i - 1; end
      fprintf('temporal resnet layers at %s\n', net.layers(i).name);
      if ~iscell(inputVar), inputVar={inputVar}; end

      if opts.injectSum

        sumVar = net.layers(i).outputs ;

        net.addLayer([name '_sum'] , ...
          dagnn.Sum(), ...
          [sumVar{:}, inputVar], ...
          [name '_sum']) ;
        inputVar = [name '_sum'] ;
      %         % sigma()
      %       net.addLayer([name '_relu'] , ...
      %         dagnn.ReLU(), ...
      %         inputVar, ...
      %         name) ;
      %       inputVar = name ;
            tmpInputVar = inputVar;
            % chain input of l that has layer as input
            for l = 1:numel(net.layers)    
              for input = net.layers(i).outputIndexes
                sel = find(isequal(net.layers(l).inputIndexes, input)) ;
                if ~isempty(sel) && ~strcmp(net.layers(l).name, [name '_sum']);
                  [net.layers(l).inputs{sel}] = deal(tmpInputVar) ;
                  inputVar = net.layers(l).outputs{1} ;
                end
              end
            end
      else
        for jj = i:numel(net.layers)
              if isa(net.layers(jj).block, 'dagnn.Sum')
                net.layers(jj).inputs = [net.layers(jj).inputs{:}, inputVar]                
                break; 
              end;
        end
      end





  end
end

end



function G3 = G3Init3D(stage)
 
  sigma = 1;
  SAMPLING_RATE = 0.5/sigma;
  N = 3*sigma;
  t = SAMPLING_RATE*[-N:N];
  C = 0.1840;
  f_size = length(t);

  f1 = -4*C*t.*(-3 + 2*t.^2).*exp(-t.^2);
  f2 = t.*exp(-t.^2);
  f3 = -4*C*(-1 + 2*t.^2).*exp(-t.^2);
  f4 = exp(-t.^2);
  f5 = -8*C*t.*exp(-t.^2);

  
  switch stage
    case 1
      G3 = single(cat(1,f1,f3,f2,f4,f3,f5,f4,f2,f4,f4));
    case 2
      G3 = single(cat(1,f4,f2,f3,f1,f4,f2,f3,f4,f2,f4));
    case 3
      G3 = single(cat(1,f4,f4,f4,f4,f2,f2,f2,f3,f3,f1));
    otherwise
      error('undefined filtering stage');
  end
end

function G2 = G2Init3D(stage)

  sigma = 1;
  SAMPLING_RATE = 0.67;
  N = 3*sigma;
  t = SAMPLING_RATE*[-N:N];

  C = (2/sqrt(3))*(2/pi)^(3/4);

  f1 = C*(2.*t.^2 - 1).*exp(-t.^2); 
  f2 = exp(-t.^2); 
  f3 = 2*C.*t.*exp(-t.^2); 
  f4 = t.*exp(-t.^2);
  
  f_size = length(t);
  
    
  switch stage
    case 1
      G2 = single(cat(1,f1,f3,f2,f3,f2,f2));
    case 2
      G2 = single(cat(1,f2,f4,f1,f2,f3,f2));
    case 3
      G2 = single(cat(1,f2,f2,f2,f4,f4,f1));
    otherwise
      error('undefined filtering stage');
  end
end

function F = init3DG3steerSet(speeds, num_directions, filter_type)

  if nargin < 3, filter_type = 'G3'; end
  

  [thetas] = initSpacetimeOrientations(speeds, num_directions);
  switch filter_type
      case 'G3'
        F = zeros(10,4, size(thetas,1), 'single'); % steering coefficients for all orientations
      case 'G2'
        F = zeros(6,3, size(thetas,1), 'single'); 
  end
  for t = 1:size(thetas,1)
    theta = thetas(t,:);
    n = [theta(1); theta(2); theta(3)]; %[u; v; w];
    n = n/norm(n);

    e1 = [1; 0; 0];
    e2 = [0; 1; 0];

    if ( abs(acos(dot(n, e1)/norm(n))) > abs(acos(dot(n, e2)/norm(n))) )
       ua = cross(n,e1);
    else
       ua = cross(n,e2);
    end

    ua = ua/norm(ua);

    ub = cross(n,ua);
    ub = ub/norm(ub);

    ua = ua';
    ub = ub';
    
    
    switch filter_type
      case 'G3'
        directions = [cos(0)*ua      + sin(0)*ub; ...
                    cos(pi/4)*ua + sin(pi/4)*ub; ...
                    cos(2*pi/4)*ua + sin(2*pi/4)*ub; ...
                    cos(3*pi/4)*ua + sin(3*pi/4)*ub] ; 

        for i = 1:size(directions,1)
          a = directions(i,1);
          b = directions(i,2);
          c = directions(i,3);

          F(:,i,t) =   [(a^3); ...
               3*(a^2)*b; ... 
               3*a*(b^2); ...
                   (b^3); ...
               3*(a^2)*c; ...
                 6*a*b*c; ...
               3*(b^2)*c; ...
               3*a*(c^2); ...
               3*b*(c^2); ...
                   (c^3)];
        end
      case 'G2'
        directions = [cos(0)*ua      + sin(0)*ub; ...
            cos(pi/3)*ua + sin(pi/3)*ub; ...
            cos(2*pi/3)*ua + sin(2*pi/3)*ub] ; 
   

        for i = 1:size(directions,1)
          a = directions(i,1);
          b = directions(i,2);
          c = directions(i,3);

          F(:,i,t) = [(a^2); ...
               2*a*b; ... 
              (b^2); ...
                  2*a*c; ...
               2*b*c; ...
                 (c^2) ];
        end
        
    end
  end
  F = reshape(F,size(F,1),[]);
end

function orientations = initSpacetimeOrientations(speeds, num_directions)
 
thetas = (2*pi/num_directions)*[0:(num_directions-1)];
 
tmp = [cos(thetas)' sin(thetas)'];
 
orientations = [0 0 1];

 
for i = 1:length(speeds)
    orientations = [orientations
                    speeds(i)*tmp ones(length(tmp),1)];
end

% --- here we default to two channels for flicker/infinite motion one may
%     want to add additional flicker channels for diagonals
orientations = [orientations;
                .5 .5 0   %  diagonal flicker
                -.5 .5 0   %  diagonal flicker
                .5 -.5 0   %  diagonal flicker
                -.5 -.5 0   %  diagonal flicker                
                1 0 0   %  horizontal flicker
                0 1 0]; %  vertical flicker
            
end
