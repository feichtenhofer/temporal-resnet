function imdb = cnn_yup2_setup_data_imgs(datasetDir, varargin)
opts.numTrain = 21;
opts.numTest = 9;
opts.randSplit = 0;
opts.flowDir = 'brox_flow_scaled';
opts.imageDir = 'jpegs_256';
opts.splitFile = []; % or '10_90_randsplit_1.txt'

[opts, varargin] = vl_argparse(opts, varargin) ;

if opts.randSplit > 0
    rng(opts.randSplit) ;
end
imdb.vidDir.static = fullfile(datasetDir, 'camera_stationary') ;
imdb.vidDir.moving = fullfile(datasetDir, 'camera_moving') ;

imdb.imageDir = opts.imageDir;
imdb.flowDir = opts.flowDir;
if ~isdir(datasetDir),  mkdir(datasetDir) ; end
if ~exist(imdb.vidDir.moving, 'dir')
    fprintf('Downloading and extracting YUP++ videos to %s ...\n', datasetDir);
    url = 'http://vision.eecs.yorku.ca/WebShare/YUP++.zip';
    extracted_files = unzip(url,datasetDir);
end
if ~exist(imdb.imageDir, 'dir')
    fprintf('Downloading and extracting YUP++ images to %s.\n This will take a while ...\n', datasetDir);
    url = 'http://ftp.tugraz.at/pub/feichtenhofer/yup++/YUP++_jpegs_256.zip';
    fn = [datasetDir filesep 'YUP++_jpegs_256.zip'];
    try
      websave(fn, url, weboptions('Timeout', Inf, 'ContentType', 'raw')) ;
      extracted_files = unzip([datasetDir filesep 'YUP++_jpegs_256.zip'],datasetDir);
    catch
      fprintf('Failed downloading and extracting YUP++ images from %s to %s.\n Please download & extract manually.\n', url, fn);
    end
end

cats = dir(imdb.vidDir.static) ;
cats = cats([cats.isdir] & ~ismember({cats.name}, {'.','..'})) ;
imdb.classes.name = {cats.name} ;
imdb.images.id = [] ;
imdb.sets = {'static_train','static_test',  'moving_train', 'moving_test'} ;
imdb.images.name = {};
imdb.images.instances = {};
imdb.images.set = [];
imdb.images.label = {};
imdb.images.nFrames = {};
imdb.images.flowScales = {};
for s = {'static','moving'}
  for ci=1:numel(cats)
    ims = dir(fullfile(imdb.vidDir.(char(s)), imdb.classes.name{ci}, '*.mp4'));
    inst = cellfun(@(x) x(1:end-4),{ims.name},'UniformOutput',false);
    imdb.images.instances = {imdb.images.instances{:}, inst{:}} ;
    ims = cellfun(@(x)fullfile(imdb.classes.name{ci},x(1:end-4)),{ims.name},'UniformOutput',false) ;
    imdb.images.name = {imdb.images.name{:}, inst{:}} ;
    imdb.images.label{end+1} = ci * ones(1,length(ims)) ;   
    for i=1:numel(inst)
      imdb.images.nFrames{end+1} = length(dir(fullfile(imdb.imageDir, inst{i}, '*.jpg') )); 
    end
    
    if ~isempty(strfind(opts.flowDir, 'scaled'))
      for i=1:numel(inst)
        scaleFile = [opts.flowDir, filesep, 'u', filesep, inst{i}, '.bin'];
        imdb.images.flowScales{end+1} =    getFlowScale(scaleFile);
      end
    end
    
    
  end
end

imdb.images.label  = cat(2, imdb.images.label{:}) ;
imdb.images.nFrames  = cat(2, imdb.images.nFrames{:}) ;

%imdb.images.set = horzcat(imdb.images.set{:}) ;
imdb.images.id = 1:numel(imdb.images.name) ;



if opts.randSplit   
  rng(opts.randSplit) ;
  trainIdx =[];
  for i = 1 : (length(imdb.images.name) / (opts.numTrain+opts.numTest))
    trainIdx = cat(2, trainIdx, randperm( opts.numTrain+opts.numTest) <= opts.numTrain);
  end
else
    trainIdx = mod(0:length(imdb.images.name)-1, opts.numTrain+opts.numTest) < opts.numTrain ;
end

selTrain = find(trainIdx) ;
selTest = setdiff(1:length(imdb.images.name), selTrain) ;

static_moving = cat(2, zeros(1,(opts.numTrain+opts.numTest)*numel(cats)), 2*ones(1,(opts.numTrain+opts.numTest)*numel(cats)));

imdb.images.set = static_moving + 1 * trainIdx +  2 * ~trainIdx;
% use split from textfile
if ~isempty(opts.splitFile)
  f = fopen(fullfile(datasetDir, opts.splitFile),'w');
  for i=1:numel(imdb.images.name)
    fprintf( f, '%s %s\n', imdb.images.name{i}, imdb.sets{ imdb.images.set(i)});
  end
  fclose(f);
end

