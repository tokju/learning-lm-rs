use crate::config::LlamaConfigJson;
use crate::tensor::Tensor;
use safetensors::SafeTensors;
pub struct LLamaParams<T> {
    // token_id to embedding lookup table
    pub embedding_table: Tensor<T>, // (vocab_size, dim)
    // decoder layer
    pub rms_att_w: Vec<Tensor<T>>, // (hidden_size, ) x layers
    pub wq: Vec<Tensor<T>>,        // (n_heads * head_size, hidden_size) x layers
    pub wk: Vec<Tensor<T>>,        // (n_kv_heads * head_size, hidden_size) x layers
    pub wv: Vec<Tensor<T>>,        // (n_kv_heads * head_size, hidden_size) x layers
    pub wo: Vec<Tensor<T>>,        // (hidden_size, n_heads * head_size) x layers
    // ffn layer
    pub rms_ffn_w: Vec<Tensor<T>>, // (hidden_size, ) x layers
    pub w_up: Vec<Tensor<T>>,      // (intermediate_size, hidden_size) x layers
    pub w_gate: Vec<Tensor<T>>,    // (intermediate_size, hidden_size) x layers
    pub w_down: Vec<Tensor<T>>,    // (hidden_size, intermediate_size) x layers
    // output
    pub rms_out_w: Tensor<T>, // (hidden_size, )
    pub lm_head: Tensor<T>,   // (vocab_size, dim)
}

impl LLamaParams<f32> {
    pub fn from_safetensors(safetensor: &SafeTensors, config: &LlamaConfigJson) -> Self {
        // todo!("实现从safetensors文件的模型参数加载");

        // Debug
        // safetensor.names().iter().for_each(|name| {
        //     println!("Available tensor: {}", name);
        // });

        let get_tensor = |name: &str| -> Tensor<f32> {
            let tensor = safetensor.tensor(name).unwrap();
            let shape = tensor.shape().to_vec();
            let data = tensor.data();
            let data: Vec<f32> = data.chunks_exact(4)
                .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                .collect();
            Tensor::new(data, &shape)
        };

        let layers = config.num_hidden_layers;

        // Helper function to get tensors for each layer
        let get_layer_tensors = |prefix: &str, suffix: &str| -> Vec<Tensor<f32>> {
            (0..layers)
                .map(|i| get_tensor(&format!("{}.{}.{}", prefix, i, suffix)))
                .collect()
        };

        Self {
            embedding_table: get_tensor("lm_head.weight"),
            rms_att_w: get_layer_tensors("model.layers", "input_layernorm.weight"),
            wq: get_layer_tensors("model.layers", "self_attn.q_proj.weight"),
            wk: get_layer_tensors("model.layers", "self_attn.k_proj.weight"),
            wv: get_layer_tensors("model.layers", "self_attn.v_proj.weight"),
            wo: get_layer_tensors("model.layers", "self_attn.o_proj.weight"),
            rms_ffn_w: get_layer_tensors("model.layers", "post_attention_layernorm.weight"),
            w_up: get_layer_tensors("model.layers", "mlp.up_proj.weight"),
            w_gate: get_layer_tensors("model.layers", "mlp.gate_proj.weight"),
            w_down: get_layer_tensors("model.layers", "mlp.down_proj.weight"),
            rms_out_w: get_tensor("model.norm.weight"),
            lm_head: get_tensor("lm_head.weight"),
        }
    }
}
