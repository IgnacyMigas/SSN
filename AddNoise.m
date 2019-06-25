function [ret] = AddNoise(input, d)

    ret = input;
    r=size(input, 1);
    c=size(input, 2);
    s=r*c;
    for i=1:size(input, 3)
        v=randperm(s, s*d)-1;
        for j=1:length(v)
            ret(mod(v(j), r)+1, fix(v(j)/r)+1, i) = ret(mod(v(j), r)+1, fix(v(j)/r)+1, i)*-1;
        end
    end

end

