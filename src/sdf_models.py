from torch import nn

class SimSDF(nn.Module):
    def __init__(self, D, num_layers=4, num_heads=8): #D是256，num_pp=128
        super(SimSDF, self).__init__()
        
        # self.lifting = nn.Sequential(FC(3, D//4, True, 'relu'), FC(D//4, D//2, True, 'relu'), FC(D//2, D, True, 'relu'))#embedding layer
        self.sim_lifting=  nn.Linear(3, D,) # FC(3, D, True)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=D,
            dim_feedforward=D,
            nhead=num_heads,
            dropout=0.1,
            batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(D,1,)
        nn.init.xavier_uniform_(self.fc.weight)
        # self.proj = nn.Tanh(self.fc)
        self.proj = self.fc 

    @property
    def device(self):
        return next(self.parameters()).device
    
    def forward(self, sdf_q):
        #qs, eq: bsz x 3
        #lines: bsz x trj_len x 3 
        #sdf_q.requires_grad=True
        #e_sdf = self.lifting(sdf_q)
        e_sdf=self.sim_lifting(sdf_q)
        e_out = self.transformer(e_sdf)
        sdf_out=self.proj(e_out)
        return sdf_out.squeeze(-1)
    