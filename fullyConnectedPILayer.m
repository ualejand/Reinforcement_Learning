classdef fullyConnectedPILayer < nnet.layer.Layer

    properties (Learnable)
        Weights
    end
    
    methods
        function obj = fullyConnectedPILayer(Weights,Name)

            % Set layer name
            obj.Name = Name;

            % Set layer description
            obj.Description = "fullyConnectedNonNegWeightLayer";
        
            % Set layer weights
            obj.Weights = Weights;
        end
        
        function Z = predict(obj, X)
            Z = fullyconnect(X, abs(obj.Weights), 0, 'DataFormat','CB');
        end
    end
end