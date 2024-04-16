%make_events create an event file
%

clear all;


LIST_BIDS = # Subject list from BIDS
LIST_STIM = # Subject List


dir_work = pwd;
dir_response = # input directory
dir_output = # output directory
dir_image = # image directory

likelihood_csv = # directory for supplementary task data
true_categ = likelihood_csv.categ1_ave;
false_categ = likelihood_csv.categ2_ave;


% mkdir(dir_output)

% estimated values
estimated_para = readtable('estimated_parameters_weighted');


% dirlist = dir('/Users/shuhei/Desktop/fmriprep-replaced_T1');
% dirlist = dir('/Users/shuhei/Desktop/workspace/heudiconv/BIDS');
dirlist = dir(dir_response);
%get indices of not subject's directory that satisfy
ix_notsubj =arrayfun(@(x) strncmp(x.name,'.',1)|strcmp(x.name,dir_output)|~x.isdir,dirlist, 'UniformOutput',false);
ix_subj = ~cell2mat(ix_notsubj);

subjlist = arrayfun(@(x) x.name, dirlist, 'UniformOutput',false);
subjlist = subjlist(ix_subj);

contlist = subjlist;
indicesToRemove = find(ismember(subjlist, {'sub-DI','sub-HM','sub-RM','sub-KH','sub-MF','sub-MOt','sub-FA','sub-KT','sub-SY','sub-TY'}));
contlist(indicesToRemove)=[];

%%Prepare image file names
exp_prefix = 'Full';
dir_exp = fullfile(dir_image,exp_prefix);
num_images = 100;
d = dir([dir_exp,'/*.jpg']);
ix_notimg = cell2mat(arrayfun(@(x) strncmp(x.name,'.',1), d, 'UniformOutput',false));
d(ix_notimg) = [];
image_filenames = {d(:).name};

%% These are fixed?
init_rest = 32;
prior_block = 4;
stimulus_block = 8;
posterior_block = 6;
%%ここら辺は確認しないといけないね
post_rest = 6;
columns = {'onset','duration','trial_type','modulation'};
% trial type prior:1, image stimulus:2, posterior:3

% likelihoodの計算
for sub=1:length(contlist)
subjid = subjlist{sub};
% convert subjid (derived from dirnames) to LIST_BIDS-style through LIST_STIM
%ix_list = strcmp(subjid, LIST_STIM);
%subjid_bids = LIST_BIDS{ix_list}; % I GUESS subjid is already bids style

dir_subj = fullfile(dir_response,subjid);

stim_files = dir(dir_subj); %honnmani?
%Remove files start from '.'
ix_notfile = cell2mat(arrayfun(@(x) strncmp(x.name,'.',1), stim_files, 'UniformOutput',false));
stim_files(ix_notfile) = [];

%The numb
% er of runs in each sessions
run_in_session = length(stim_files); % Assume only 1 session/subj

%Prepare output directory
dir_output_subj = fullfile(dir_work,dir_output,subjid);

for r = 1:5 %5回分のrunを回している
    stim_file = fullfile(dir_subj,stim_files(r).name);
    
    %Convert stim order and response files

%     fprintf('Loading %s\n', stim_file);
    stim = load(stim_file);

    event_matrix = {};

    %Stimulus blocks ???
    stim_prior = stim.StimPriorRun;
    stim_posterior = stim.PosteriorProb;
    stim_image = stim.ImgSeq;
    stim_corloc = stim.CorLoc;
    
%     stim_correct_rate(13, 1)
%     cell2mat(stim_correct_rate(13, 2))
%     find(stim_correct_rate(:,2),68)
    
%     Z = cellfun(@(x)reshape(x,1,1,[]),stim_correct_rate,'un',0);
%     out = cell2mat(Z)
%     find(out,68)
%     [row, col]=find(out==68,1)
%     out(Index)
%     size(out)
%     shuhei


    for i = 1:20
        if stim_corloc(i,1) == 1
            stim_correct_rate{i+20*(r-1),1} = stim_posterior(i);
            stim_correct_rate{i+20*(r-1),2} = stim_image(i);
        elseif stim_corloc(i,1) == 2
            stim_correct_rate{i+20*(r-1),1} = 100 - stim_posterior(i);
            stim_correct_rate{i+20*(r-1),2} = stim_image(i);
        end
    end
    
%     matchingRows = find(stim_correct_rate(:,2)==5);
    
    Z = cellfun(@(x)reshape(x,1,1,[]),stim_correct_rate,'un',0);
    out = cell2mat(Z);

end


for img = 1:100
    [row, col]=find(out==img,1);
    image_info{img,sub} = out(row,1);
end
end

% cell2mat(image_info)


ave_ima = mean(cell2mat(image_info),2);



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



for allsub=1:length(subjlist)
subjid = subjlist{allsub};

dir_subj = fullfile(dir_response,subjid);

stim_files = dir(dir_subj); %honnmani?
%Remove files start from '.'
ix_notfile = cell2mat(arrayfun(@(x) strncmp(x.name,'.',1), stim_files, 'UniformOutput',false));
stim_files(ix_notfile) = [];

%The numb
% er of runs in each sessions
run_in_session = length(stim_files); % Assume only 1 session/subj

%Prepare output directory
dir_output_subj = fullfile(dir_work,dir_output,subjid);
mkdir(dir_output_subj)

% parameter
sub_index = find(strcmp(estimated_para.Var1,subjid(5:end)));

% wp = estimated_para{sub_index,2};
% ws = estimated_para{sub_index,3};
% alpha_p = estimated_para{sub_index,4};
% alpha_s = estimated_para{sub_index,5};
wp=1;
ws=1;
alpha_p=0;
alpha_s=0;

for r = 1:5 %5回分のrunを回している
    stim_file = fullfile(dir_subj,stim_files(r).name);
    
    %Convert stim order and response files

    fprintf('Loading %s\n', stim_file);
    stim = load(stim_file);

    event_matrix = {};

    %Stimulus blocks ???
    stim_prior = stim.StimPriorRun;
    stim_posterior = stim.PosteriorProb;
    stim_image = stim.ImgSeq;
    stim_corloc = stim.CorLoc;

    %Prepare for prior
%     opara_pri = max(cell2mat(stim_prior),[],2) - 70;

    %Prepare for likelihood
    for i = 1:20
        if stim_corloc(i,1) == 1
            stim_prior_cor(i) = stim_prior{i}(1);
        elseif stim_corloc(i,1) == 2
            stim_prior_cor(i) = stim_prior{i}(2);
        end
    end

    corrected_prior = reshape(stim_prior_cor,[20,1]);
%     for i = 1:20
%         if corrected_prior(i) == 90
%             prior_label(i) = "High_Prior";
%         elseif corrected_prior(i) == 10
%             prior_label(i) = "High_Prior";
%         elseif corrected_prior(i) == 70
%             prior_label(i) = "Middle_Prior";
%         elseif corrected_prior(i) == 30
%             prior_label(i) = "Middle_Prior";
%         elseif corrected_prior(i) == 50
%             prior_label(i) = "Low_Prior";
%         end
%     end
%     prior_label_cell = cellstr(prior_label);

    Cp = corrected_prior/100;
    Fp = 1-Cp;
    Lp = log(Cp ./ Fp);
%     aplp = alpha_p .* Lp;
%     Faplpwp = log((wp.*exp(aplp)+1-wp) ./ ((1-wp).*exp(aplp)+wp));
    Faplpwp=0;
    

    log_prior_pre = log(stim_prior_cor);
    log_prior = reshape(log_prior_pre,[20,1]);
    log_ave = log(ave_ima(stim_image) ./ (100 - ave_ima(stim_image)));
    stim_likelihood = log_ave - log_prior;


    Ls = stim_likelihood;
%     asls = Ls .* alpha_s;
%     Faslsws = log((ws.*exp(asls)+1-ws) ./ ((1-ws).*exp(asls)+ws));
    Faslsws=0;
    
    argument_prior = Lp+Faslsws+Faplpwp;
    argument_likelihood = Ls+Faplpwp+Faslsws;
    
    prior_list = log((wp.*exp(argument_prior+1-wp)) ./ ((1-wp).*exp(argument_prior)+wp));
    likelihood_list = log((ws.*exp(argument_likelihood)+1-ws) ./ ((1-ws).*exp(argument_likelihood)+ws));

    opara_pri = prior_list - mean(prior_list);
    opara_lik = likelihood_list - mean(likelihood_list);
% 
%     %Prepare for posterior
%     posterior_para = max(stim_posterior,[],2);
%     opara_pos = posterior_para - mean(posterior_para);


    for i = 1:20 % それぞれのrun
        %Prior
        event_matrix{3*i-2,1} = 32+ 18 * (i-1);
        event_matrix{3*i-2,2} = prior_block;
        event_matrix{3*i-2,3} = 'IM_Prior';
%         event_matrix(3*i-2,3) = prior_label_cell(i);
        event_matrix{3*i-2,4} = opara_pri(i) / 20; %This value is for left stimuli
%         event_matrix{3*i-2,4} = log((wp.*exp(argument_prior(i))+1-wp) ./ ((1-wp).*exp(argument_prior(i))+wp));
        %Likelihood
        event_matrix{3*i-1,1} = 36+ 18 * (i-1);
        event_matrix{3*i-1,2} = stimulus_block;
        event_matrix{3*i-1,3} = 'IM_Likelihood';
        event_matrix{3*i-1,4} = opara_lik(i) / max(opara_lik);
%         event_matrix{3*i-1,4} = log((ws.*exp(argument_likelihood(i))+1-ws) ./ ((1-ws).*exp(argument_likelihood(i))+ws));
        %Posterior
        event_matrix{3*i,1} = 44+18*(i-1);
        event_matrix{3*i,2} = posterior_block;
        event_matrix{3*i,3} = 'Posterior';
        event_matrix{3*i,4} = event_matrix{3*i-2,4} + event_matrix{3*i-1,4};

    end
    
    %Save the event file
    output_file = fullfile(dir_output_subj, ...
        sprintf('%s_task-bayes_run-0%02d_events.tsv',subjid,r));
    fid = fopen(output_file, 'w');
    fprintf(fid, '%s\t%s\t%s\t%s\n',columns{:});
    for i = 1:size(event_matrix,1)
        %Convert NaN -> n/a
        write_str = sprintf('%d\t%d\t%s\t%d\n', event_matrix{i, :});
        write_str = strrep(write_str, 'NaN', 'n/a');
            
        fprintf(fid, '%s', write_str);
    end
end


end




