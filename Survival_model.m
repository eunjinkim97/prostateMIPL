clc
clear
close all
%% Features load
path = 'dl_result/';
fold_num = [350, 349, 349, 350, 350];
for fold_idx = 1:5
    fold_name = strcat(num2str(fold_idx-1),'fold');
    data = readtable(strcat(fold_name, '_train_960features.csv'));
    
    data(1,:) = [];
    data(:,1:2)= [];
    data(1:5,1:5)
    data.Properties.VariableNames(1:3)={'ID','BCR','date'};

    tr_size = fold_num(fold_idx); %(350, 349, 349, 350, 350)
    cv=table([true(tr_size,1);false(size(data,1)-tr_size,1)]);
    data_cv=[cv data];
    data_cv.Properties.VariableNames(1) = {'CV_training'};

    cli=readtable('clinical_variables.xlsx');
    cli=removevars(cli, {'modality'});
    clinicals_name = cli.Properties.VariableNames(2:8);
    clinicals=cli(:,2:8); % 1 ID, 2~73 rf, 74~end clinics
    inner=innerjoin(data_cv,clinicals);

    bcr=inner(:,3);
    bcr_arr=bcr{:,:};
    bcr_m=inner(:,4); 
    bcr_m=bcr_m{:,:};
    inner.Properties.VariableNames(1:5)

    td=inner(:,1);
    td=td{:,:};
    CV.training = td;
    CV.test = ~td;

    %% Features
    data = inner;
    features = data(:,5:end);% 5 ~ 25092 : DL feature, 25093 ~ 25163 : rf, 25164 ~ end : clinical feature
    totalFeatures=features;
    features_name = totalFeatures.Properties.VariableNames;
    totalFeatures=table2array(totalFeatures);

    % z-normalization
    TrainFeatures = totalFeatures(CV.training,:);
    TrainClass = bcr_arr(CV.training,:);
    TrainLabel = [bcr_m(CV.training,:),~bcr_arr(CV.training,:)]; % recurrence 0 (1 censored)
    TestFeatures = totalFeatures(CV.test,:);
    TestClass = bcr_arr(CV.test,:);
    TestLabel = [bcr_m(CV.test,:),~bcr_arr(CV.test,:)]; % recurrence 0 (1 censored)

    [zTrain, mu, sig] = zscore(TrainFeatures);
    zTest = (TestFeatures-mu)./sig;
    TrainFeatures = zTrain; TestFeatures=zTest;

    ver_load=1;
    result = struct();
    for i=1:2
        if i==1 % radiomics
            save_path = strcat(path, fold_name);
            tr = TrainFeatures(:,1:960);
            te = TestFeatures(:,1:960);
            disp(size(tr))
            disp(size(te))

            % Lasso Cox
            if ver_load==0
                cox_lasso = cvglmnet(tr,TrainLabel,'cox');
                cvglmnetPlot(cox_lasso)
                saveas(gcf, strcat(save_path,'/survival/Lassofitting_', num2str(i),'ver.png'));
                save(strcat(save_path,'/survival/variable_coxlasso.mat'),'-struct','cox_lasso')
                selectedFeature_idx = find(cox_lasso.glmnet_fit.beta(:,(cox_lasso.lambda == cox_lasso.lambda_min)));
                while isempty(selectedFeature_idx)==1
                    close all
                    cox_lasso = cvglmnet(tr,TrainLabel,'cox');
                    cvglmnetPlot(cox_lasso)
                    saveas(gcf, strcat(save_path,'/survival/Lassofitting_', num2str(i),'ver.png'));
                    save(strcat(save_path,'/survival/variable_coxlasso.mat'),'-struct','cox_lasso')
                    selectedFeature_idx = find(cox_lasso.glmnet_fit.beta(:,(cox_lasso.lambda == cox_lasso.lambda_min)));
                end
            elseif ver_load==1
                cox_lasso=struct();
                cox_lasso=load(strcat(save_path,'/survival/variable_coxlasso.mat'));
                selectedFeature_idx = find(cox_lasso.glmnet_fit.beta(:,(cox_lasso.lambda == cox_lasso.lambda_min)));
            end    

        end

        result(i).selectedFeatureName = features_name(:,selectedFeature_idx);
        result(i).selectedFeature_coef =cox_lasso.glmnet_fit.beta(selectedFeature_idx,(cox_lasso.lambda == cox_lasso.lambda_min));

        selectedFeature = tr(:,selectedFeature_idx);
        selectedFeature_te = te(:,selectedFeature_idx);
        disp(size(selectedFeature))


        if i==2
            clinical=TrainFeatures(:,961:966);
            clinical_te=TestFeatures(:,961:966);
            selectedFeature = [selectedFeature, clinical];
            selectedFeature_te = [selectedFeature_te, clinical_te];
            disp(size(selectedFeature))
            disp(size(selectedFeature_te))
        end

    % Training fitting
        [b,logl,H,stats] = coxphfit(selectedFeature, TrainLabel(:,1),'Censoring',TrainLabel(:,2));
        result(i).b_tr = b;result(i).rad_score = selectedFeature * b;
        result(i).tr_m_rad_pfs = median(result(i).rad_score);
        [lambda1, lambda2, result(i).trHR, result(i).trHRci, UL, SUL, z, result(i).p_pfs_tr, alpha] = logrank([TrainLabel(result(i).rad_score>=result(i).tr_m_rad_pfs,1),TrainLabel(result(i).rad_score>=result(i).tr_m_rad_pfs,2)],[TrainLabel(result(i).rad_score<result(i).tr_m_rad_pfs,1),TrainLabel(result(i).rad_score<result(i).tr_m_rad_pfs,2)]);
        saveas(gcf, strcat(save_path,'/DL_Training_KM_', num2str(i),'ver.png'));

        % Test fitting
        [b_te,logl,H,stats_te] = coxphfit(selectedFeature_te, TestLabel(:, 1),'Censoring',TestLabel(:,2)); 
        result(i).b_te = b_te;
        result(i).rad_score_te = selectedFeature_te *b_te;
        result(i).te_m_rad_pfs = median(result(i).rad_score_te);

        [lambda1, lambda2, result(i).teHR, result(i).teHRci, UL, SUL, z, result(i).p_pfs_te, alpha] = logrank([TestLabel(result(i).rad_score_te>=result(i).te_m_rad_pfs,1),TestLabel(result(i).rad_score_te>=result(i).te_m_rad_pfs,2)], [TestLabel(result(i).rad_score_te<result(i).te_m_rad_pfs,1), TestLabel(result(i).rad_score_te<result(i).te_m_rad_pfs,2)]);
        saveas(gcf, strcat(save_path,'/DL_Test_KM_', num2str(i),'ver.png'));

        result(i).trHRci = round(result(i).trHRci,2);
        result(i).teHRci = round(result(i).teHRci,2);
        save(strcat(save_path,'/DL_results.mat'),'result')

        % risk score saving
        dce_risk_tr=[table(result(i).rad_score),table(TrainLabel)];
        writetable(dce_risk_tr,strcat(save_path,'/DL_riskscore_tr_', num2str(i),'ver.xlsx'))
        dce_risk=[table(result(i).rad_score_te),table(TestLabel)];
        writetable(dce_risk,strcat(save_path,'/DL_riskscore_te_', num2str(i),'ver.xlsx'))

    end
    
end