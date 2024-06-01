% number of realizations to generate
N = 1500;

% parameters for the Gaussian random field
gamma = 4;
tau = 5;
sigma = 25^(2);

% viscosity
visc = 0.01;

% grid size
s = 4096;
steps = 100; % 这里表示的是(0,1)时间步长内的选点数目
steps = 1; 
nn = 101;

% input = zeros(N, nn);
input = zeros(N, s);
if steps == 1
    output = zeros(N, s);
else
   output = zeros(N, steps, nn);
   output = zeros(N, steps, s)
end

tspan = linspace(0,1,steps+1);
x = linspace(0,1, s+1);
X = linspace(0,1, nn);

for j=1:N
    u0 = GRF(s/2, 0, gamma, tau, sigma, "periodic");
    u = Burgers(u0, tspan, s, visc);
    
    % u0_eval = u0(X);
    u0_eval = u0(x);
    input(j,:) = u0_eval(1:end-1);
    
    if steps == 1
        output(j,:) = u.values;
    else
        for k=1:(steps+1)
            output(j,k,:) = u{k}(X);
        end
    end
    
    disp(j);
end
save('Burger.mat', 'input', 'output', 'tspan',  'gamma', 'tau', 'sigma')

