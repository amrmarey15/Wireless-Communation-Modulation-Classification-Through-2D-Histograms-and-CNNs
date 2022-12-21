function g = QAM_generate(M)
% This function generate QAM signals 
g = 0                                         ;
switch(M) 
    case 8,
        index1 = 0 ; s = 0                     ;
        for x = -3:2:3
            for y = -1:2:1
                index1 = index1 + 1            ;
                g(index1) = x + j*y            ;
                s = s + abs(g(index1))^2       ;
            end 
        end 
        k = M/s                                ;
        g = g *sqrt(k)                         ;
        
    case 16,
        index1 = 0 ; s = 0                     ;
        for x = -3:2:3
            for y = -3:2:3
                index1 = index1 + 1            ;
                g(index1) = x + j*y            ;
                s = s + abs(g(index1))^2       ;
            end 
        end 
       k = M/s                                ;
       g = g *sqrt(k)                         ;
        
     case 32,
        index1 = 0 ; s = 0                     ;
        for x = -7:2:7
            for y = -3:2:3
                index1 = index1 + 1            ;
                g(index1) = x + j*y            ;
                s = s + abs(g(index1))^2       ;
            end 
        end 
        k = M/s                                ;
        g = g *sqrt(k)                         ;
        
        case 64,
        index1 = 0 ; s = 0                     ;
        for x = -7:2:7
            for y = -7:2:7
                index1 = index1 + 1            ;
                g(index1) = x + j*y            ;
                s = s + abs(g(index1))^2       ;
            end 
        end 
        k = M/s                                ;
        g = g *sqrt(k)                         ;   
end
        