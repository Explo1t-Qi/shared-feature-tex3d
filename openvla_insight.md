# OpenVLA

## 核心目标要能回答5个问题

1. 模型由哪些部分组成
2. 论文中的模块与仓库中代码的对应关系
3. 数据流完整调用链
4. 训练和推理分别在哪里发生
5. 如果要修改代码，添加模块，应该如何修改


## 阅读的 4 个层次

1. 仓库地图，不管具体实现，知道模块的大致作用
2. 推理层：沿一条数据流贯穿，从 LIBERO 中得到一帧 RGB 后，如何得到机器人的 action，
3. 训练层: training sample -> loss
4. OpenVLA 内部实现：Vision Encoder/Projector/LLM/Action decoder

## 预期产出

1. 模型结构图
2. 仓库调用结构
3. OpenVLA -> oft 演化

## 阅读原则

1. 按真实请求的数据流追踪
2. 忽略与模型行为无关的代码
3. 明确每个 API 所在的层级
4. 论文与代码同步


## 开始阅读

### Phase 1

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-08/ab2530c0-a1af-47fd-a4af-446004ba722e/4494010b12883291ff2987171110715f87813682ac7f102f32d96b279e55c5bb.jpg)

即输入 Image 和 Language 输出 Action

```
Image
  │
  ├──→ DINOv2 ──┐
  │              ├── concat
  └──→ SigLIP ──┘
                  │
                  ↓
             2-layer MLP
              Projector
                  │
                  ↓
Language ─────→ Llama 2 7B
                  │
                  ↓
            Action Tokens
                  │
                  ↓
             7D Action
```

从 GitHub clone 到本地的仓库目录，可以看到其中有 prismatic 模块，这主要是以 [Prismatic](https://github.com/TRI-ML/prismatic-vlms) VLM 为基座，在其中略作修改:

```
➜  openvla git:(main) tree
.
├── LICENSE
├── Makefile
├── README.md
├── experiments
│   └── robot
│       ├── bridge ...
│       ├── libero
│       │   ├── libero_requirements.txt
│       │   ├── libero_utils.py
│       │   ├── regenerate_libero_dataset.py
│       │   └── run_libero_eval.py     --> 一次完整的 libero 运行参考
│       ├── openvla_utils.py
│       └── robot_utils.py
├── prismatic             --> 需要重点掌握的层次
│   ├── __init__.py
│   ├── conf
│   │   ├── __init__.py
│   │   ├── datasets.py
│   │   ├── models.py
│   │   └── vla.py        --> 新增与 openvla 相关
│   ├── extern
│   │   ├── __init__.py
│   │   └── hf
│   │       ├── __init__.py
│   │       ├── configuration_prismatic.py
│   │       ├── modeling_prismatic.py
│   │       └── processing_prismatic.py
│   ├── models    --> 模型结构
│   │   ├── __init__.py
│   │   ├── backbones
│   │   │   ├── __init__.py
│   │   │   ├── llm
│   │   │   │   ├── __init__.py
│   │   │   │   ├── base_llm.py
│   │   │   │   ├── llama2.py        --> LLM 处理部分，把 hugging face 的 llmam 包装成 prismatic 所需的 LLMBackbone
│   │   │   │   ├── mistral.py
│   │   │   │   ├── phi.py
│   │   │   │   └── prompting
│   │   │   │       ├── __init__.py
│   │   │   │       ├── base_prompter.py
│   │   │   │       ├── llama2_chat_prompter.py
│   │   │   │       ├── mistral_instruct_prompter.py
│   │   │   │       ├── phi_prompter.py
│   │   │   │       └── vicuna_v15_prompter.py
│   │   │   └── vision
│   │   │       ├── __init__.py
│   │   │       ├── base_vision.py
│   │   │       ├── clip_vit.py
│   │   │       ├── dinoclip_vit.py
│   │   │       ├── dinosiglip_vit.py    --> dino + siglip 拼接，论文中的 dino_v2 + siglip
│   │   │       ├── dinov2_vit.py
│   │   │       ├── in1k_vit.py
│   │   │       └── siglip_vit.py
│   │   ├── load.py
│   │   ├── materialize.py
│   │   ├── registry.py
│   │   ├── vlas
│   │   │   ├── __init__.py
│   │   │   └── openvla.py       --> 新增的 openvla 模块，但是实际上就是一个简单的包装
│   │   └── vlms
│   │       ├── __init__.py
│   │       ├── base_vlm.py
│   │       └── prismatic.py     --> 把视觉和 LLM 拼起来(Vision Encoder + Porjector -> LLM)
│   ├── overwatch
│   │   ├── __init__.py
│   │   └── overwatch.py
│   ├── preprocessing
│   │   ├── __init__.py
│   │   ├── datasets
│   │   │   ├── __init__.py
│   │   │   └── datasets.py
│   │   ├── download.py
│   │   └── materialize.py
│   ├── py.typed
│   ├── training
│   │   ├── __init__.py
│   │   ├── materialize.py
│   │   ├── metrics.py
│   │   └── strategies
│   │       ├── __init__.py
│   │       ├── base_strategy.py
│   │       ├── ddp.py
│   │       └── fsdp.py
│   ├── util
│   │   ├── __init__.py
│   │   ├── batching_utils.py
│   │   ├── data_utils.py
│   │   ├── nn_utils.py
│   │   └── torch_utils.py
│   └── vla
│       ├── __init__.py
│       ├── action_tokenizer.py       -->  动作离散化 token，编解码器
│       ├── datasets
│       │   ├── __init__.py
│       │   ├── datasets.py
│       │   └── rlds
│       │       ├── __init__.py
│       │       ├── dataset.py
│       │       ├── obs_transforms.py
│       │       ├── oxe
│       │       │   ├── __init__.py
│       │       │   ├── configs.py
│       │       │   ├── materialize.py
│       │       │   ├── mixtures.py
│       │       │   ├── transforms.py
│       │       │   └── utils
│       │       │       └── droid_utils.py
│       │       ├── traj_transforms.py
│       │       └── utils
│       │           ├── __init__.py
│       │           ├── data_utils.py
│       │           ├── goal_relabeling.py
│       │           └── task_augmentation.py
│       └── materialize.py
├── pyproject.toml
├── requirements-min.txt
├── scripts
│   ├── additional-datasets
│   │   ├── lrv_instruct.py
│   │   └── lvis_instruct_4v.py
│   ├── extern
│   │   ├── convert_prismatic_weights_to_hf.py
│   │   └── verify_prismatic.py
│   ├── generate.py
│   ├── preprocess.py
│   └── pretrain.py
└── vla-scripts    --> 训练入口层
    ├── deploy.py
    ├── extern
    │   ├── convert_openvla_weights_to_hf.py
    │   └── verify_openvla.py
    ├── finetune.py    --> lora 微调
    └── train.py      --> VLA 训练，全量微调



```


#### libero 一次数据完整的传输

```python
@draccus.wrap()  
def eval_libero(cfg: GenerateConfig) -> None:  
	...
    # Load model  
    model = get_model(cfg)  # 使用 huggingface api 初始化 openvla 模型(神经网络)
  
	...  
    # [OpenVLA] Get Hugging Face processor  
    processor = None  
    if cfg.model_family == "openvla":  
        processor = get_processor(cfg)  # 将 image+text 变成模型接受的 tensor
        
        
    ...
    
    while t < max_steps + cfg.num_steps_wait:  
    try:  
        # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects  
        # and we need to wait for them to fall        
        if t < cfg.num_steps_wait:  
            obs, reward, done, info = env.step(get_libero_dummy_action(cfg.model_family))  
            t += 1  
            continue  
  
        # Get preprocessed image  
        img = get_libero_image(obs, resize_size)  
        # get_libero_image 中实际发生的就是下面三句代码
        # img = obs["agentview_image"] 获取模拟器环境中的图像，然后转一百八十度，缩放成 224 * 224 的可接收格式
        # img = img[::-1, ::-1]  # IMPORTANT: rotate 180 degrees to match train preprocessing
        # img = resize_image(img, resize_size) 224 * 224*
  
        # Save preprocessed image for replay video  
        replay_images.append(img)  
  
        # Prepare observations dict  
        # Note: OpenVLA does not take proprio state as input        
        observation = {  
            "full_image": img,  # 整个 observation 真正使用到的只有 full_image 字段
            "state": np.concatenate(  
                (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])  
            ),  # 这里是更通用的 robot policy 接口，但是 openvla 并不会使用
        }  
  
        # Query model to get action  
        action = get_action(  
            cfg,  
            model,  
            observation,  
            task_description,  
            processor=processor,  
        )  # 最后真正调用的是 action = vla.predict_action(**inputs, unnorm_key=unnorm_key, do_sample=False)
        
        # Normalize gripper action [0,1] -> [-1,+1] because the environment expects the latter  
		action = normalize_gripper_action(action, binarize=True)  # 将归一化数据转化成标准化数据
```

`get_action` 实际上包装的方法是 `get_vla_action`，在调用方法后验证输出的形状 `assert action.shape == (ACTION_DIM,)` ：

```python
def get_vla_action(vla, processor, base_vla_name, obs, task_label, unnorm_key, center_crop=False):  
    """Generates an action with the VLA policy."""  
    image = Image.fromarray(obs["full_image"])  
    image = image.convert("RGB")  
  
    # (If trained with image augmentations) Center crop image and then resize back up to original size.  
    # IMPORTANT: Let's say crop scale == 0.9. To get the new height and width (post-crop), multiply    
    #            the original height and width by sqrt(0.9) -- not 0.9!    
    if center_crop:  
        batch_size = 1  
        crop_scale = 0.9  
  
        # Convert to TF Tensor and record original data type (should be tf.uint8)  
        image = tf.convert_to_tensor(np.array(image))  
        orig_dtype = image.dtype  
  
        # Convert to data type tf.float32 and values between [0,1]  
        image = tf.image.convert_image_dtype(image, tf.float32)  
  
        # Crop and then resize back to original size  
        image = crop_and_resize(image, crop_scale, batch_size)  
  
        # Convert back to original data type  
        image = tf.clip_by_value(image, 0, 1)  
        image = tf.image.convert_image_dtype(image, orig_dtype, saturate=True)  
  
        # Convert back to PIL Image  
        image = Image.fromarray(image.numpy())  
        image = image.convert("RGB")  
  
    # Build VLA prompt  
    if "openvla-v01" in base_vla_name:  # OpenVLA v0.1  
        prompt = (  
            f"{OPENVLA_V01_SYSTEM_PROMPT} USER: What action should the robot take to {task_label.lower()}? ASSISTANT:"  
        )  
    else:  # OpenVLA  
        prompt = f"In: What action should the robot take to {task_label.lower()}?\nOut:"  
  
    # Process inputs.  
    inputs = processor(prompt, image).to(DEVICE, dtype=torch.bfloat16)  # 将输入数据变成神经网络接受的 tensor  
    # Get action.    
    action = vla.predict_action(**inputs, unnorm_key=unnorm_key, do_sample=False)  # 真正进入 VLA 模型
    # 实际上调用的是 openvla/prismatic/extern/hf/modeling_prismatic.py: OpenVLAForActionPrediction.predict_action 这个方法
    return action
```

此处即为深入 VLA 模型的分界线，processor 控制输入 tensor 的生成，predict_action 是真正进入 VLA 神经网络：

```
────────────────────────────────

LIBERO / preprocessing world

image
language
processor

────────────────────────────────
              ↓
      vla.predict_action()
────────────────────────────────

OpenVLA model world

vision encoder
projector
LLM
action tokens
decode
unnormalize

────────────────────────────────
```

整体输入流程如下:

```
                LIBERO
                   │
                   ↓
           obs["agentview_image"]
                   │
          rotate + resize
                   │
                   ↓
             224×224 RGB
                   │
                   │
task.language ─────┤
                   ↓
          OpenVLA prompt
                   │
                   ↓
     processor(prompt, image)
                   │
                   ↓
        pixel_values + tokens
                   │
                   ↓
          vla.predict_action()
                   │
                   ↓
              7D action
                   │
                   ↓
       gripper postprocessing
                   │
                   ↓
            env.step(action)
                   │
                   ↓
               new obs
                   │
                   └──────────────↺
```

### Phase 2

这里主要探究第一个 action token 如何得到，后续的 action token 主要根据 KV Cache 得到。
#### predict_action

承接上面的 vla.predict_action 实际调用的是 prismatic/extern/hf/modeling_prismatic.py: predict_action

首先一个需要的基础信息是 llama 7B 词表长度是 32000，openvla 没有新增动作头，而是将词表最后的低频词表示动作，真正使用的 action token ids 大致是 31744-31999 共 256 个：

```
Llama vocabulary

0
│
│ 普通 language tokens
│
│
31743
├────────────────────
31744  ← action token
31745
31746
 ...
31998
31999  ← action token
├────────────────────
32000
```

```python
class OpenVLAForActionPrediction(PrismaticForConditionalGeneration):  
    config_class: PretrainedConfig = OpenVLAConfig  
  
    def __init__(self, config: OpenVLAConfig) -> None:  
        super().__init__(config)  
        self.norm_stats = config.norm_stats  
  
        # Compute action bins  
        self.bins = np.linspace(-1, 1, config.n_action_bins)   # 可以看到 bins 是 -1, 1 之前均匀取 256 份，即 255 次分割
        self.bin_centers = (self.bins[:-1] + self.bins[1:]) / 2.0  # 分割出 256 份，这里看似是人为划分出来的，但实际上只要训练时和推理时一致，模型就不会出错
  
        # Compute vocab size for de-tokenization -- revert added "multiple of"  
        self.vocab_size = self.config.text_config.vocab_size - self.config.pad_to_multiple_of  
  
    def predict_action(  
        self, input_ids: Optional[torch.LongTensor] = None, unnorm_key: Optional[str] = None, **kwargs: str  
    ) -> np.ndarray:  
        """Thin wrapper around .generate() that decodes predicted actions and unnormalizes them."""  
        # If the special empty token ('') does not already appear after the colon (':') token in the prompt  
        # (after "OUT:" or "ASSISTANT:"), insert it to match the inputs seen at training time        
        if not torch.all(input_ids[:, -1] == 29871):  # 推理时需要补充一个空白 token，保证和 openvla 训练时的格式一致
            input_ids = torch.cat(  
                (input_ids, torch.unsqueeze(torch.Tensor([29871]).long(), dim=0).to(input_ids.device)), dim=1  
            )  
  
        # Run VLA inference  真正调用 openvla 进行推理
        generated_ids = self.generate(input_ids, max_new_tokens=self.get_action_dim(unnorm_key), **kwargs)  # get_action_dim 获取到的维度长度就是 7
  
        # Extract predicted action tokens and translate into (normalized) continuous actions  
        # generated_ids 在输入 VLA 后，generate 方法在 input_ids 后加上新生成的 tokens
        predicted_action_token_ids = generated_ids[0, -self.get_action_dim(unnorm_key) :].cpu().numpy()  # 可以看到这里获取了 generated_ids 的最后 7 位动作 token
        discretized_actions = self.vocab_size - predicted_action_token_ids  # 这里是 openvla 的约定: token id 小，对应的 bin 大。即浮点数值 -1 ... 1 对应的 bin index 为 0 ... 255，对应的 token id 为 31999 ... 31744
        discretized_actions = np.clip(discretized_actions - 1, a_min=0, a_max=self.bin_centers.shape[0] - 1)  
        normalized_actions = self.bin_centers[discretized_actions]  # 现在得到标准化后的数值
  
        # Unnormalize actions  
        action_norm_stats = self.get_action_stats(unnorm_key)    # 不同的 dataset，action_norm_stats 动作分布不同，所以需要单独获取
        mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["q01"], dtype=bool))  # 有的 action 需要有的不需要，所以这里要带上 mask
        action_high, action_low = np.array(action_norm_stats["q99"]), np.array(action_norm_stats["q01"]) # 选择 q99 和 q01 而不是 min/max，避免极少数极端动作把 normalization range 拉得过大
        actions = np.where(  
            mask,  
            0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,  # 归一化公式逆变换
            normalized_actions,  
        )  
  
        return actions
```

输入经过动作预测后如何变成向量值:

```
input_ids
pixel_values
    │
    │
    ↓
self.generate(...)     --> 下一步重点
    │
    │  max_new_tokens = 7
    ↓
generated_ids
    │
    ├── prompt token IDs
    │
    └── 7 action token IDs
              │
              ↓
       取最后 7 个
              │
              ↓
     [t1, t2, ..., t7]
              │
              ↓
       vocab_size - token
              │
              ↓
          bin index
              │
              ↓
          bin center
              │
              ↓
  normalized_actions ∈ [-1,1]^7
              │
              ↓
          q01 / q99
              │
              ↓
       unnormalization
              │
              ↓
      continuous action ∈ R^7
```

这里也就能看出为什么 llama 可以控制机器人动作，llama 本是自回归模型，根据前 n 个输出预测 n+1 个输出。openvla 将词表的最后 256 个 token 赋予了动作意义，模型正常预测，只是后面这些 token 表示机器人的动作。这里的经典 OpenVLA 动作预测仍然是自回归，这是与 OFT 最大的结构变化之一。


#### self.generate 生成第一个 action token

向上找引用可以找到 `...\lib\python3.10\site-packages\transformers\generation\utils.py`，这个方法属于 GenerationMixin 这个类:

```python
class GenerationMixin:  
    """  
    A class containing all functions for auto-regressive text generation, to be used as a mixin in [`PreTrainedModel`].  
    The class exposes [`~generation.GenerationMixin.generate`], which can be used for:        
    - *greedy decoding* by calling [`~generation.GenerationMixin._greedy_search`] if `num_beams=1` and          `do_sample=False`        
    - *contrastive search* by calling [`~generation.GenerationMixin._contrastive_search`] if `penalty_alpha>0` and          `top_k>1`        
    - *multinomial sampling* by calling [`~generation.GenerationMixin._sample`] if `num_beams=1` and          `do_sample=True`        
    - *beam-search decoding* by calling [`~generation.GenerationMixin._beam_search`] if `num_beams>1` and          `do_sample=False`        
    - *beam-search multinomial sampling* by calling [`~generation.GenerationMixin._beam_sample`] if `num_beams>1`          and `do_sample=True`        
    - *diverse beam-search decoding* by calling [`~generation.GenerationMixin._group_beam_search`], if `num_beams>1`          and `num_beam_groups>1`        
    - *constrained beam-search decoding* by calling [`~generation.GenerationMixin._constrained_beam_search`], if          `constraints!=None` or `force_words_ids!=None`        
    - *assisted decoding* by calling [`~generation.GenerationMixin._assisted_decoding`], if            `assistant_model` or `prompt_lookup_num_tokens` is passed to `.generate()`  
    You do not need to call any of the above methods directly. Pass custom parameter values to 'generate' instead. To    learn more about decoding strategies refer to the [text generation strategies guide](../generation_strategies).    """  
    def prepare_inputs_for_generation(self, *args, **kwargs):  
        raise NotImplementedError(  
            "A model class needs to define a `prepare_inputs_for_generation` method in order to use `.generate()`."  
        )
```

所以说必须要实现 prepare_inputs_for_generation 这个方法，而 prismatic/extern/hf/modeling_prismatic.py 中也确实实现了。这里解码时会有各个方法，它们其中都调用了 forward 方法，所以可以看到 modeling_prismatic.py 里也实现了一份 forward：

```
predict_action()
        ↓
generate()
        ↓
prepare_inputs_for_generation()
        ↓
forward(
    input_ids,
    pixel_values,
    attention_mask
)
        ↓
    第一次完整
 multimodal forward
        ↓
      logits
        ↓
选择下一个 token
        ↓
第 1 个 action token
```

`prismatic/extern/hf/modeling_prismatic.py: class PrismaticForConditionalGeneration`:

```python
# === Core Prismatic VLM `forward()` Logic ===  
def forward(  
    self,  
    input_ids: Optional[torch.LongTensor] = None,  
    attention_mask: Optional[torch.Tensor] = None,  
    pixel_values: Optional[torch.FloatTensor] = None,  
    labels: Optional[torch.LongTensor] = None,  
    inputs_embeds: Optional[torch.FloatTensor] = None,  
    past_key_values: Optional[List[torch.FloatTensor]] = None,  
    use_cache: Optional[bool] = None,  
    output_attentions: Optional[bool] = None,  
    output_hidden_states: Optional[bool] = None,  
    output_projector_features: Optional[bool] = None,  
    return_dict: Optional[bool] = None,  
) -> Union[Tuple, PrismaticCausalLMOutputWithPast]:  
    """Run a forward pass through the VLM, returning a PrismaticCausalLMOutputWithPast instance."""  
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions  
    output_hidden_states = (  
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states  
    )  
    output_projector_features = output_projector_features if output_projector_features is not None else False  
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict  
  
    # Respect `use_cache` only if not training (even if `gradient_checkpointing` is off)  
    use_cache = use_cache and not self.training  
  
    # Instantiate Placeholder for Projector Features  
    projected_patch_embeddings = None  
  
    # Note :: We only support forward passes with the following cases:  
    #   => Cached Generation :: (input_ids.shape[1] == 1) and (past_key_values is not None)    
    #   => Unimodal Forward :: (pixel_values is None)    
    #   => Multimodal Forward :: (pixel_values is not None) and (input_ids/embeds.shape[0] == pixel_values.shape[0])  
    # === Handle Generation with Cache (`input_ids.shape[1] == 1`) =>> requires `past_keys_values` ===    
    ... 
  
    # === Handle Multimodal Forward ===  根据输入的数据类型判断是否是多模态，前面还有 Cache 和 单模态的处理逻辑，这里直接突出重点
    elif (input_ids.shape[0] == pixel_values.shape[0]) or (inputs_embeds.shape[0] == pixel_values.shape[0]):  
        assert past_key_values is None, "Unexpected key `past_key_values` provided during language-only forward!"  
  
        # Visual Feature Extraction  
        patch_features = self.vision_backbone(pixel_values)  # 1. pixel values 前面处理过，是 [1, 6, 224, 224] dino + siglip 这里就先处理图像，得到小图像块组成的 patch_features
  
        # Projection Logic =>> Update Attention Mask  
        projected_patch_embeddings = self.projector(patch_features)  # 将图像进行投影，准备与文本一同送入到 llama 中进行处理
        projected_patch_attention_mask = None  
        if attention_mask is not None:  
            projected_patch_attention_mask = torch.full(  
                (projected_patch_embeddings.shape[0], projected_patch_embeddings.shape[1]),  
                fill_value=True,  
                dtype=attention_mask.dtype,  
                device=attention_mask.device,  
            )  
  
        # Get Input Embeddings (from Language Model Embeddings)  
        input_embeddings = self.get_input_embeddings()(input_ids)  
  
        # Build Multimodal Embeddings & Attention Mask =>> Prismatic defaults to inserting after <BOS> token (1:)  
        multimodal_embeddings = torch.cat(  
            [input_embeddings[:, :1, :], projected_patch_embeddings, input_embeddings[:, 1:, :]], dim=1  
        )  
        multimodal_attention_mask = None  
        if attention_mask is not None:  
            multimodal_attention_mask = torch.cat(  
                [attention_mask[:, :1], projected_patch_attention_mask, attention_mask[:, 1:]], dim=1  
            )  
  
        # Build Labels (if specified) =>> Ignore Labels for Patch Embeddings  
        multimodal_labels = None  
        if labels is not None:  
            projected_patch_labels = torch.full(  
                (projected_patch_embeddings.shape[0], projected_patch_embeddings.shape[1]),  
                fill_value=IGNORE_INDEX,  
                dtype=labels.dtype,  
                device=labels.device,  
            )  
            multimodal_labels = torch.cat([labels[:, :1], projected_patch_labels, labels[:, 1:]], dim=1)  
  
        # Dispatch to Language Model  
        language_model_output = self.language_model(  
            input_ids=None,  
            attention_mask=multimodal_attention_mask,  
            position_ids=None,  
            past_key_values=None,  
            inputs_embeds=multimodal_embeddings,  
            labels=multimodal_labels,  
            use_cache=use_cache,  
            output_attentions=output_attentions,  
            output_hidden_states=output_hidden_states,  
            return_dict=return_dict,  
        )  
  
    # === Otherwise =>> Assume Invalid! ===  
    elif (input_ids.shape[0] != pixel_values.shape[0]) or (inputs_embeds.shape[0] != pixel_values.shape[0]):  
        raise ValueError("Non-homogenous batch of (text, image) input -- forward() does not support mixed batches!")  
  
    else:  
        raise ValueError(  
            "Invalid PrismaticForConditionalGeneration `forward()` call with provided arguments:\n"  
            f"=> `input_ids` = {input_ids is not None}\n"  
            f"=> `attention_mask` = {attention_mask is not None}\n"  
            f"=> `pixel_values` = {pixel_values is not None}\n"  
            f"=> `labels` = {labels is not None}\n"  
            f"=> `input_embeds` = {inputs_embeds is not None}\n"  
            f"=> `past_key_values` = {past_key_values is not None}\n"  
            f"=> `use_cache` = {use_cache}"  
        )  
  
    # Unpack `language_model_output` and return PrismaticCausalLMOutputWithPast (or tuple if not `return_dict`)  
    if not return_dict:  
        if output_projector_features and (projected_patch_embeddings is not None):  
            return *language_model_output, projected_patch_embeddings  
  
        return language_model_output  
  
    return PrismaticCausalLMOutputWithPast(  
        loss=language_model_output.loss,  
        logits=language_model_output.logits,  
        past_key_values=language_model_output.past_key_values,  
        hidden_states=language_model_output.hidden_states,  
        attentions=language_model_output.attentions,  
        projector_features=projected_patch_embeddings,  
    )
```

```python

# 该类的构造方法中初始化了如下几个关键变量

# Instantiate PrismaticVisionBackbone (w/ Potential Fused Backbone)  
self.vision_backbone = PrismaticVisionBackbone(  
    config.use_fused_vision_backbone, config.image_sizes, config.timm_model_ids, config.timm_override_act_layers  
)  
  
# Create Multimodal Projector  
self.projector = PrismaticProjector(  
    config.use_fused_vision_backbone,  
    vision_dim=self.vision_backbone.embed_dim,  
    llm_dim=config.text_config.hidden_size,  
)  
  
# Instantiate LLM Backbone  
self.language_model = AutoModelForCausalLM.from_config(  
    config.text_config, attn_implementation=config._attn_implementation  
)
```

下面逐个来看调用

##### PrismaticVisionBackbone.forward

具体来看 `patch_features = self.vision_backbone(pixel_values)` 的处理逻辑：

```python
# === Prismatic Vision Backbone (nn.Module) Definitions (w/ Fused Backbone Support) ===  
class PrismaticVisionBackbone(nn.Module):  
    def __init__(  
        self,  
        use_fused_vision_backbone: bool,  
        image_sizes: List[int],  
        timm_model_ids: List[str],  
        timm_override_act_layers: List[Optional[str]],  
    ) -> None:  
        super().__init__()  
        self.use_fused_vision_backbone = use_fused_vision_backbone  
  
        # [Contract] Validate number of (fused) vision backbones, create "alpha" featurizer and Instantiate  
        #   =>> Note :: Monkey-Patch the `forward()` function of the backbone to ensure FSDP-compatibility        
        #               Hardcodes `get_intermediate_layers` to return the **SECOND-TO-LAST** layer patches!        
        assert len(timm_model_ids) <= 2, "Prismatic models only support up to 2 (fused) vision backbones!"  
        self.featurizer = timm.create_model(  
            timm_model_ids[0],  # 配置文件中为 vit_large_patch14_reg4_dinov2.lvd142m
            pretrained=False,  
            num_classes=0,  
            img_size=image_sizes[0],  
            act_layer=timm_override_act_layers[0],  
        )  
        self.featurizer.forward = unpack_tuple(  
            partial(self.featurizer.get_intermediate_layers, n={len(self.featurizer.blocks) - 2})  # 这里获取的是倒数第二个 transformer block 的 patch features
        )  
        self.embed_dim = self.featurizer.embed_dim  
  
        # If `use_fused_vision_backbone` =>> create "beta" featurizer  
        if self.use_fused_vision_backbone:  # 配置文件中为 true
            self.fused_featurizer = timm.create_model(  
                timm_model_ids[1],  # 配置文件中为 vit_so400m_patch14_siglip_224
                pretrained=False,  
                num_classes=0,  
                img_size=image_sizes[1],  
                act_layer=timm_override_act_layers[1],  
            )  
            self.fused_featurizer.forward = unpack_tuple(  
                partial(self.fused_featurizer.get_intermediate_layers, n={len(self.fused_featurizer.blocks) - 2})  # 这里获取的是倒数第二个 transformer block 的 patch features
            )  
            self.embed_dim += self.fused_featurizer.embed_dim  
  
        # Patch `vision_backbone.featurizer` and `vision_backbone.fused_featurizer` with HF-Compatible LayerScale  
        for module in self.featurizer.modules():  
            if isinstance(module, LayerScale):  
                ls_apply_patch(module)  
  
        if self.use_fused_vision_backbone:  
            for module in self.fused_featurizer.modules():  
                if isinstance(module, LayerScale):  
                    ls_apply_patch(module)  
  
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:  
        """Run image (`pixel_values`) through featurizer; if channel-stacked, then dispatch and sequence stack."""  
        if not self.use_fused_vision_backbone:  
            return self.featurizer(pixel_values)  
  
        # Split `pixel_values :: [bsz, 2 * 3, resolution, resolution]` =>> featurize =>> channel stack  
        img, img_fused = torch.split(pixel_values, [3, 3], dim=1)  # 先将拼接的图像信息拆开，得到 DINO input [1,3,224,224] 和 SigLIP input [1,3,224,224]
        patches, patches_fused = self.featurizer(img), self.fused_featurizer(img_fused)  # vision transformer 实际处理图像的位置
  
        return torch.cat([patches, patches_fused], dim=2)
```

通过上面对 feature forward 的初始化，可以看到这里获取的是倒数第二个 transformer block 的 patch features

```
             同一个 RGB image
                  │
          ┌───────┴───────┐
          ↓               ↓
       DINOv2           SigLIP
          │               │
          ↓               ↓
     DINO patches    SigLIP patches
```

论文中写的是 256 个 patch：两个视觉模型输入均为 224 * 224，输入 patch size 均为 14 * 14，所以一共 16 * 16 = 256 个 patch
`vit_large_patch14_reg4_dinov2.lvd142m` 的向量长度是 1024，而 `vit_so400m_patch14_siglip_224` 的向量长度是 1152，所以当 batch_size = 1 时：

```
DINOv2
[1, 256, 1024]

SigLIP
[1, 256, 1152]
```

最后 forward 方法在维度上进行拼接，得到 `[1,256,1024+1152]` 形状的 tensor:

```python
return torch.cat([patches, patches_fused], dim=2)
```

一起使用两个 ViT 是因为 DINOv2 侧重于形状/几何上的理解，而 SigLIP 更偏向于视觉与语义上的对齐。这样 OpenVLA 希望同时得到视觉结构+语言语义信息。

##### PrismaticProjector.foward

因为视觉 embeddings 是要进入 llama 的，而 llama 7B hidden dimension 长度是 4096，与 2176 不同，所以需要 Projector 进行投影。

```python
# === Prismatic Projector (nn.Module) Definitions ===  
class PrismaticProjector(nn.Module):  
    def __init__(self, use_fused_vision_backbone: bool, vision_dim: int, llm_dim: int) -> None:  
        super().__init__()  
        self.use_fused_vision_backbone = use_fused_vision_backbone  
        self.vision_dim, self.llm_dim = vision_dim, llm_dim  
  
        # Switch on `use_fused_vision_backbone` =>> use slightly different MLPs and projection factors!  
        if not self.use_fused_vision_backbone:  
            self.fc1 = nn.Linear(self.vision_dim, self.llm_dim, bias=True)  
            self.fc2 = nn.Linear(self.llm_dim, self.llm_dim, bias=True)  
            self.act_fn1 = nn.GELU()  
        else:  
            initial_projection_dim = 4 * vision_dim  
            self.fc1 = nn.Linear(self.vision_dim, initial_projection_dim, bias=True)  
            self.fc2 = nn.Linear(initial_projection_dim, self.llm_dim, bias=True)  
            self.fc3 = nn.Linear(self.llm_dim, self.llm_dim, bias=True)  
            self.act_fn1 = nn.GELU()  
            self.act_fn2 = nn.GELU()  
  
    def forward(self, img_patches: torch.Tensor) -> torch.Tensor:  
        if not self.use_fused_vision_backbone:  
            projected_features = self.fc1(img_patches)  
            projected_features = self.act_fn1(projected_features)  
            projected_features = self.fc2(projected_features)  
        else:  
            projected_features = self.fc1(img_patches)  
            projected_features = self.act_fn1(projected_features)  
            projected_features = self.fc2(projected_features)  
            projected_features = self.act_fn2(projected_features)  
            projected_features = self.fc3(projected_features)  
  
        return projected_features
```

这里是自己实现的简单网络，这里看 fused backbone:

```
2176
 ↓ Linear
8704 (2176 * 4)
 ↓ GELU
4096
 ↓ GELU
4096
```

##### self.get_input_embeddings

简单来说就是获取 llama 7B 的 embedding table:

```python
# === `PreTrainedModel` Boilerplate ===  
def get_input_embeddings(self) -> nn.Module:  
    return self.language_model.get_input_embeddings()

...

input_embeddings = self.get_input_embeddings()(input_ids)
```

`input_ids` 有 T 个，那么此处获取到的  `input_embeddings` 就有 T 个，形状是 \[T, 4096\] ，毕竟 llama 中每个 token 对应的 embeddings 维度是 4096


##### 拼接多模态输入，准备送入 llama 处理

```python

# 还是 forward 方法

# Build Multimodal Embeddings & Attention Mask =>> Prismatic defaults to inserting after <BOS> token (1:)  
multimodal_embeddings = torch.cat(  
    [input_embeddings[:, :1, :], projected_patch_embeddings, input_embeddings[:, 1:, :]], dim=1  
)  
multimodal_attention_mask = None  
if attention_mask is not None:  
    multimodal_attention_mask = torch.cat(  
        [attention_mask[:, :1], projected_patch_attention_mask, attention_mask[:, 1:]], dim=1  
    )
```

其实这里的 cat 操作如下，这里的注意力掩码也是类似的操作：

```
原来：

[BOS] [text1] [text2] [text3] ... [textT]


插入视觉 token：


[BOS]
   ↓
[IMAGE_1]
[IMAGE_2]
...
[IMAGE_256]
   ↓
[text1]
[text2]
...
[textT]
```

##### llama 处理输入

```python
# 还是 forward 方法

# Dispatch to Language Model  
language_model_output = self.language_model(  
    input_ids=None,  
    attention_mask=multimodal_attention_mask,  
    position_ids=None,  
    past_key_values=None,  
    inputs_embeds=multimodal_embeddings,  
    labels=multimodal_labels,  
    use_cache=use_cache,  
    output_attentions=output_attentions,  
    output_hidden_states=output_hidden_states,  
    return_dict=return_dict,  
)
```

以下可以想象为 llama 获取到的信息，视觉和语义关系在训练时均已确定

```
Llama sequence

position
   0       BOS

   1       image patch 1
   2       image patch 2
   ...
   256     image patch 256

   257     "In"
   258     ":"
   259     "What"
   260     "action"
   ...
   N       "Out:"
```

最后得到输出：

```python

# forward 方法

return PrismaticCausalLMOutputWithPast(  
    loss=language_model_output.loss,  
    logits=language_model_output.logits,  
    past_key_values=language_model_output.past_key_values,  
    hidden_states=language_model_output.hidden_states,  
    attentions=language_model_output.attentions,  
    projector_features=projected_patch_embeddings,  
)
```

llama 最后的 LM head 把 4096 维 hidden state 映射到词表 32000 维 logits。

另外在调用时的传参是:

```python
# experiments/robot/openvla_utils.py

# Get action.  
action = vla.predict_action(**inputs, unnorm_key=unnorm_key, do_sample=False)
```

`do_sample=False` 对应的 greedy decoding (self.generate 提到的 GenerationMixin 中有注释表明)，所以使用 argmax 。第一个 action token 就是对 LM head 最后输出取 argmax 得到。

> Llama 内部先把 `[1,256+T,4096]` 的 hidden states 通过 LM head 转成 `[1,256+T,V]` 的 logits；`generate()` 只取最后一个 sequence position 的 `[1,V]` logits，在整个 vocabulary 中选出下一个 token。由于 OpenVLA 已经针对 action-token prediction 训练，所以正常情况下最高分会落在 action-token 区域，但这不是代码层面的硬约束。
> 
> 这里的 T 是语言 token 长度，V 是整个词表的长度，真正在处理时是 32000+64，为了提高计算效率而加的 padding，最后会减去，仍然是 32000


整体数据流图如下:

```
RGB image
   │
   ↓ Processor
pixel_values
[1, 6, 224, 224]
   │
   ├──────────────────────────┐
   ↓                          ↓
DINOv2                     SigLIP
[1,256,1024]              [1,256,1152]
   │                          │
   └────────────┬─────────────┘
                ↓ concat(feature dim)
        [1,256,2176]
                │
                ↓ Projector
        [1,256,4096]
                │
                │
input_ids       │
[1,T]           │
   ↓            │
Llama Embedding │
[1,T,4096]      │
   └───────┬────┘
           ↓
[BOS | 256 image embeddings | text embeddings]

[1,T+256,4096]
           │
           ↓
       Llama-2 7B
           │
           ↓
 hidden states
[1,T+256,4096]
           │
           ↓ LM Head
        logits
[1,T+256,V]
           │
           ↓
 last-position logits
        [1,V]
           │
        argmax
           ↓
     action token 1
```


#### self.generate 生成后续 action token

与上面处理逻辑不同的是，如果 past_key_values 不为空的话，就不再把整个 prompt 送入 llama，而是只保留最后一个没有处理过的 token：

```python
# === GenerationMixin Methods ===  
def prepare_inputs_for_generation(  
    self,  
    input_ids: Optional[torch.Tensor] = None,  
    past_key_values: Optional[List[torch.FloatTensor]] = None,  
    inputs_embeds: Optional[torch.FloatTensor] = None,  
    pixel_values: Optional[torch.FloatTensor] = None,  
    attention_mask: Optional[torch.Tensor] = None,  
    **kwargs: str,  
) -> Dict[str, torch.Tensor]:  
    """Borrowed from `LlamaForCausalLM` and simplified for batch size = 1; mirrors original PrismaticVLM logic."""  
    ...
  
    # Handle `past_key_values` (cache) =>> assume `input_ids` just has unprocessed tokens  
    if past_key_values is not None:  
        input_ids = input_ids[:, -1:]  # 这里就只有最后一个未被处理过的 token
  
    ...
  
    # Make sure `pixel_values` are preserved in `model_inputs`  
    model_inputs.update(  
        {  
            "attention_mask": attention_mask,  
            "pixel_values": pixel_values,  
            "past_key_values": past_key_values,  
            "use_cache": kwargs.get("use_cache"),  
        }  
    )  
  
    return model_inputs
```

此时再看 forward 有一个 `input_ids.shape[1] == 1` 的逻辑，正好就是对应上面的处理。这代表不是第一次 multimodal forward 而是做后续 autoregressive decoding：

```python
# === Core Prismatic VLM `forward()` Logic ===  
def forward(  
    self,  
    input_ids: Optional[torch.LongTensor] = None,  
    attention_mask: Optional[torch.Tensor] = None,  
    pixel_values: Optional[torch.FloatTensor] = None,  
    labels: Optional[torch.LongTensor] = None,  
    inputs_embeds: Optional[torch.FloatTensor] = None,  
    past_key_values: Optional[List[torch.FloatTensor]] = None,  
    use_cache: Optional[bool] = None,  
    output_attentions: Optional[bool] = None,  
    output_hidden_states: Optional[bool] = None,  
    output_projector_features: Optional[bool] = None,  
    return_dict: Optional[bool] = None,  
) -> Union[Tuple, PrismaticCausalLMOutputWithPast]:  
    """Run a forward pass through the VLM, returning a PrismaticCausalLMOutputWithPast instance."""  
    ...
  
    # Note :: We only support forward passes with the following cases:  
    #   => Cached Generation :: (input_ids.shape[1] == 1) and (past_key_values is not None)    #   => Unimodal Forward :: (pixel_values is None)    #   => Multimodal Forward :: (pixel_values is not None) and (input_ids/embeds.shape[0] == pixel_values.shape[0])  
    # === Handle Generation with Cache (`input_ids.shape[1] == 1`) =>> requires `past_keys_values` ===    
    
    if input_ids.shape[1] == 1:  
        assert input_ids.shape[0] == 1, "Generation is only currently supported for batch size of 1!"  
        assert past_key_values is not None, "You must provide `past_key_values` during cached generation!"  
        assert labels is None, "Unexpected key `labels` provided during cached generation!"  
		
		# 可以看到这里不再后 vision_backbone 和 projector，只是
		
        language_model_output = self.language_model(  
            input_ids=input_ids,  
            attention_mask=None,  
            position_ids=None,  
            past_key_values=past_key_values,  
            inputs_embeds=None,  
            labels=None,  
            use_cache=use_cache,  
            output_attentions=output_attentions,  
            output_hidden_states=output_hidden_states,  
            return_dict=return_dict,  
        )
```

已有第一个动作 token，所以下一个流程如此，当前 action 的第二个基于第一个生成：

```
                 KV Cache
          Image + Language
                 │
                 │
t₁ ──────────────┤
                 ↓
              Llama
                 ↓
              logits
                 ↓ 
              argmax
                 ↓
                 t₂
```


这里有一个很细的时间关系。第一次： $C=(I,L)$ 得到：t1​ 。此时 cache 是：$KV(C)$ 注意这时候：t1​ **还没有加入 cache**，因为 t1​ 是第一次 forward **结束以后才选出来的**。因此第二次 forward past_key_values = $KV(C)$  
input token = t₁

Llama 会：
1. 给 t1​ 计算新的 Q/K/V；
2. 用 Qt1​​ 去 attention: $K_C​+K_{t_1}$​​
3. 生成预测 t2​ 的 hidden state；
4. 同时把: $K_{t_1}​​,V_{t_1}​$​ 加到 cache。

第二次结束后 cache 变成: $KV(C,t_1​)​$ 这点很重要。后续 token 皆是如此。比如 step3 时，输入 $t_2$ 加上 $KV(C,t_1)$ 得到 $t_3$

```python
══════════════ Step 1 ══════════════

input:
    image
    language

Vision Backbone
      ↓
Projector
      ↓
Llama(full sequence)
      ↓
logits
      ↓
t₁

save:
KV(image + language)


══════════════ Step 2 ══════════════

input:
    only t₁

cache:
    KV(image + language)

Llama
  ↓
t₂

save:
KV(image + language + t₁)


══════════════ Step 3 ══════════════

input:
    only t₂

cache:
    KV(image + language + t₁)

Llama
  ↓
t₃


            ...


══════════════ Step 7 ══════════════

input:
    only t₆

cache:
    KV(image + language
       + t₁+t₂+t₃+t₄+t₅)

Llama
  ↓
t₇
```

已经完成数据的同步，可以让 codex 查看了



### Phase 3

`processor` 到底做了什么?

```python
    # Process inputs.  
    inputs = processor(prompt, image).to(DEVICE, dtype=torch.bfloat16)  # 将输入数据变成神经网络接受的 tensor  
```

`prismatic/extern/hf/processing_prismatic.py` 具体调用此处：

```python
# === PrismaticProcessor =>> Wraps both ImageProcessor and Tokenizer ===  
#   =>> https://github.com/huggingface/transformers/blob/main/src/transformers/models/llava/processing_llava.py  
class PrismaticProcessor(ProcessorMixin):  
    attributes: ClassVar[List[str]] = ["image_processor", "tokenizer"]  
    image_processor_class: str = "AutoImageProcessor"  
    tokenizer_class: str = "AutoTokenizer"  
  
    def __init__(  
        self,  
        image_processor: Optional[ImageProcessingMixin] = None,  
        tokenizer: Optional[PreTrainedTokenizerBase] = None,  
    ) -> None:  
        super().__init__(image_processor, tokenizer)  
  
    def __call__(  
        self,  
        text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]],  
        images: Union[Image.Image, List[Image.Image]],  
        padding: Union[bool, str, PaddingStrategy] = False,  
        truncation: Optional[Union[bool, str, TruncationStrategy]] = None,  
        max_length: Optional[int] = None,  
        return_tensors: Optional[Union[str, TensorType]] = TensorType.PYTORCH,  
    ) -> BatchFeature:  
        """  
        Preprocess a given (batch) of text/images for a Prismatic VLM; forwards text to the underlying LLM's tokenizer,        forwards images to PrismaticImageProcessor.        
        @param text: The (batch) of text to encode; must be a string or list of strings.        
        @param images: A (batch of) PIL.Image.Image instance(s) to preprocess.        
        @param padding: Sequence padding strategy (if multiple specified) in < True = "longest" | "max_length" | False >        
        @param truncation: Truncation strategy for the output sequences; requires `max_length` to be specified        
        @param max_length: Maximum length (in tokens) to truncate        
        @param return_tensors: Type of return tensors (usually "pt" or TensorType.PYTORCH)        
        @return: BatchFeature with keys for `input_ids`, `attention_mask` and `pixel_values`.        """        
        pixel_values = self.image_processor(images, return_tensors=return_tensors)["pixel_values"]  
        text_inputs = self.tokenizer(  
            text, return_tensors=return_tensors, padding=padding, truncation=truncation, max_length=max_length  
        )  
  
        # [Validate] Need same number of images and text inputs!  
        if pixel_values.shape[0] != text_inputs.input_ids.shape[0]:  
            raise ValueError("Batch is malformed; expected same number of images and text inputs!")  
  
        return BatchFeature(data={**text_inputs, "pixel_values": pixel_values})  
  
    # === Tokenizer Dispatch Utilities =>> check `PreTrainedTokenizerBase` for documentation ===  
    def batch_decode(  
        self,  
        sequences: Union[List[int], List[List[int]], torch.Tensor, Any],  # `Any` = np.ndarray | tf.Tensor  
        skip_special_tokens: bool = False,  
        clean_up_tokenization_spaces: Optional[bool] = None,  
        **kwargs: str,  
    ) -> List[str]:  
        return self.tokenizer.batch_decode(  
            sequences=sequences,  
            skip_special_tokens=skip_special_tokens,  
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,  
            **kwargs,  
        )  
  
    def decode(  
        self,  
        token_ids: Union[int, List[int], torch.Tensor, Any],  # `Any` = np.ndarray | tf.Tensor  
        skip_special_tokens: bool = False,  
        clean_up_tokenization_spaces: Optional[bool] = None,  
        **kwargs: str,  
    ) -> str:  
        return self.tokenizer.decode(  
            token_ids=token_ids,  
            skip_special_tokens=skip_special_tokens,  
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,  
            **kwargs,  
        )  
  
    @property  
    def model_input_names(self) -> List[str]:  
        tokenizer_input_names = self.tokenizer.model_input_names  
        image_processor_input_names = self.image_processor.model_input_names  
  
        return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names))
```

因为在 `experiments/robot/openvla_utils.py` 中已经注册完毕：

```python
def get_vla(cfg):  
    """Loads and returns a VLA model from checkpoint."""  
    # Load VLA checkpoint.  
    print("[*] Instantiating Pretrained VLA model")  
    print("[*] Loading in BF16 with Flash-Attention Enabled")  
  
    # Register OpenVLA model to HF Auto Classes (not needed if the model is on HF Hub)  
    AutoConfig.register("openvla", OpenVLAConfig)  
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)  
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)  
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)
    ...
```

```
PrismaticProcessor
│
├── image_processor
│      └── PrismaticImageProcessor
│
└── tokenizer
       └── LlamaTokenizer
```

`BatchFeature` 就是一个包装类，其实什么都没有，真正的就是下面这个字典，字典解包后涵盖的键就包括 `input_ids`/`attention_mask`/`pixel_values`：

```python
return BatchFeature(data={**text_inputs, "pixel_values": pixel_values}) 
```

`input_ids` 和 `attention_mask` 都是 tokenizer 解析得到。prompt 被 tokenizer 切出多少 token，input_ids 长度就为多少。

```
                 processor(prompt, image)
                         │
              ┌──────────┴──────────┐
              ↓                     ↓
           prompt                  image
              │                     │
              ↓                     ↓
       LlamaTokenizer       PrismaticImageProcessor
              │                     │
              ↓                     ↓
         input_ids             pixel_values
      attention_mask
              │                     │
              └──────────┬──────────┘
                         ↓
                    BatchFeature
```

这里还不会出现 llama 中 4096 维的向量， processor 只到 token IDs 这一步：

```
processor
────────────────────────
token IDs
────────────────────────
model
────────────────────────
Embedding Table
↓
4096-d embeddings
```

然后可以看到在 `PrismaticProcessor` 中调用 `__call__` 时对图像的处理实际是，所以需要看 `PrismaticImageProcessor` 的 `__call__`  方法：
```python
pixel_values = self.image_processor(images, return_tensors=return_tensors)["pixel_values"]
```

```python
class PrismaticImageProcessor(ImageProcessingMixin):  
    model_input_names: ClassVar[List[str]] = ["pixel_values"]  
  
    def __init__(  
        self,  
        use_fused_vision_backbone: bool = False,  
        image_resize_strategy: str = "letterbox",  
        input_sizes: Optional[List[Tuple[int, int, int]]] = None,  
        interpolations: Optional[List[str]] = None,  
        means: Optional[List[Tuple[float, float, float]]] = None,  
        stds: Optional[List[Tuple[float, float, float]]] = None,  
        **kwargs: str,  
    ) -> None:  
        """  
        Initialize a PrismaticImageProcessor as a wrapper around a torchvision transform; this transform will be        
        created by TIMM, and edited to follow our custom `image_resize_strategy` logic.        
        @param use_fused_vision_backbone: Boolean indicating single or fused (dual) vision backbone        
        @param image_resize_strategy: Prismatic image resize strategy in < resize-naive | resize-crop | letterbox >        
        @param input_size: [TIMM :: `data_cfg`] Input image size as tuple (channels, width, height)        
        @param interpolation: [TIMM :: `data_cfg`] Interpolation as string (default: "bicubic")        
        @param mean: [TIMM :: `data_cfg`] Normalization mean as float tuple (or two-tuple if `fused_backbone`)        
        @param std: [TIMM :: `data_cfg`] Normalization std as float tuple (or two-tuple if `fused_backbone`)        
        """        
        self.use_fused_vision_backbone = use_fused_vision_backbone  
        self.image_resize_strategy = image_resize_strategy  
  
        # Handle `None` default values  
        input_sizes = [(3, 224, 224)] if input_sizes is None else input_sizes  
        means = [(0.5, 0.5, 0.5)] if means is None else means  
        stds = [(0.5, 0.5, 0.5)] if stds is None else stds  
  
        # TIMM `data_cfg` Parameters  
        self.input_sizes, self.interpolations, self.means, self.stds = input_sizes, interpolations, means, stds  
  
        # Grab torchvision transforms via TIMM =>> need to parse for specific "functional" transform values!  
        self.tvf_resize_params, self.tvf_crop_params, self.tvf_normalize_params = [], [], []  
        self.tvf_do_letterbox, self.tvf_letterbox_fill = False, None  
  
        for idx in range(len(input_sizes)):  
            transform = timm.data.create_transform(  
                input_size=self.input_sizes[idx],  
                interpolation=self.interpolations[idx],  
                mean=self.means[idx],  
                std=self.stds[idx],  
                crop_pct=1.0,  # Set to 1.0 to ignore cropping (initial Resize sets `input_size`)  
                crop_mode="center",  # Default crop mode -- no-op when `crop_pct == 1.0`  
                is_training=False,  # No image augmentations when loading the transform!  
            )  
  
            # [Validation] Ensure appropriate transform structure, expected sizes  
            if not (  
                isinstance(transform, Compose)  
                and (len(transform.transforms) == 4)  
                and isinstance(transform.transforms[0], Resize)  
                and isinstance(transform.transforms[1], CenterCrop)  
                and isinstance(transform.transforms[2], ToTensor)  
                and isinstance(transform.transforms[3], Normalize)  
                and (transform.transforms[0].size == self.input_sizes[idx][-1])  
                and (transform.transforms[1].size == self.input_sizes[idx][-2:])  
            ):  
                raise ValueError(f"Unexpected TIMM image transformation structure/sizes: `{transform}`")  
  
            # HF Image Processors *must* be JSON-serializable; as such, cannot have torchvision. as an attribute.  
            #   => Instead, we're going to parse the transform and call "torchvision.transforms.functional" (`tvf`)            resize_t, crop_t, norm_t = transform.transforms[0], transform.transforms[1], transform.transforms[3]  
            self.tvf_resize_params.append(  
                {  
                    "size": resize_t.size,  
                    "interpolation": TVF.pil_modes_mapping[resize_t.interpolation],  
                    "max_size": None,  
                    "antialias": True,  
                }  
            )  
            self.tvf_crop_params.append({"output_size": crop_t.size})  
            self.tvf_normalize_params.append(  
                {  
                    "mean": norm_t.mean.float().numpy().tolist(),  
                    "std": norm_t.std.float().numpy().tolist(),  
                    "inplace": False,  
                }  
            )  
            self.tvf_do_letterbox, self.tvf_letterbox_fill = False, None  
  
            # Handle Prismatic `image_resize_strategy`  
            if self.image_resize_strategy == "resize-naive":  
                self.tvf_resize_params[idx]["size"] = (resize_t.size, resize_t.size)  
            elif self.image_resize_strategy == "letterbox":  
                self.tvf_do_letterbox, self.tvf_letterbox_fill = True, tuple([int(x * 255) for x in self.means[idx]])  
            elif self.image_resize_strategy == "resize-crop":  
                pass  
            else:  
                raise ValueError(f"Image resize strategy `{self.image_resize_strategy}` is not supported!")  
  
        # Dispatch **kwargs to super()  
        super().__init__(**kwargs)  
  
    def apply_transform(self, img: Image.Image) -> torch.Tensor:  
        """Apply `functional` variant of TIMM's Transform = Compose([Resize -> CenterCrop -> ToTensor -> Normalize])"""  
        if self.tvf_do_letterbox:  
            img = letterbox_pad_transform(img, self.tvf_letterbox_fill)  
  
        # [Contract] Fused Backbones expect "channel-stacked" inputs; we'll unpack on the model side!  
        imgs_t = []  
        for idx in range(len(self.input_sizes)):  
            img_idx = TVF.resize(img, **self.tvf_resize_params[idx])  
            img_idx = TVF.center_crop(img_idx, **self.tvf_crop_params[idx])  
            img_idx_t = TVF.to_tensor(img_idx)  
            img_idx_t = TVF.normalize(img_idx_t, **self.tvf_normalize_params[idx])  
            imgs_t.append(img_idx_t)  
  
        # [Contract] `imgs_t` is a list of Tensors of shape [3, input_size, input_size]; stack along dim = 0  
        img_t = torch.vstack(imgs_t)  
  
        return img_t  
  
    def preprocess(  
        self,  
        images: Union[Image.Image, List[Image.Image]],  
        return_tensors: Optional[Union[str, TensorType]] = None,  
        **_: str,  
    ) -> BatchFeature:  
        """  
        Preprocess an image (or batch of images); note that unlike the `transformers :: BaseImageProcessor` we        
        explicitly only handle PIL.Image.Image instances for simplicity.        
        @param images: A (batch of) PIL.Image.Image instance(s) to preprocess.        
        @param return_tensors: BatchFeature default Tensor format (e.g., "pt" for torch); if None, returns np.ndarray        
        @return: Instance of `transformers :: BatchFeature` with a single key "pixel_values"        
        """        
        if not isinstance(images, list):  
            images = [images]  
  
        # Apply `self.img_transform` to each image (will return list of torch.Tensors); stack into "batched" Tensor  
        pixel_values = torch.stack([self.apply_transform(img.convert("RGB")) for img in images])  
  
        # Return BatchFeature =>> note that for compatibility, constructor expects Dict[str, np.ndarray], so we convert  
        return BatchFeature(data={"pixel_values": pixel_values.float().numpy()}, tensor_type=return_tensors)  
  
    def __call__(self, images: Union[Image.Image, List[Image.Image]], **kwargs) -> BatchFeature:  
        return self.preprocess(images, **kwargs)
```

可以看到调用链条如下

```
self.image_processor -> __call__ -> self.preprocess(images, **kwargs) -> self.apply_transform(img.convert("RGB"))
```

在 `self.apply_transform` 中的处理逻辑概括如下：

```
PIL RGB
  ↓
Resize
  ↓
Center Crop
  ↓
ToTensor
  ↓
Normalize
```

其实是对同一张 RGB image 进行两套处理，它们在 normalize 时的 mean/std 值不一样：

```
                    同一张 RGB image
                           │
              ┌────────────┴────────────┐
              ↓                         ↓
       DINO preprocessing        SigLIP preprocessing
              │                         │
              ↓                         ↓
       Tensor [3,224,224]       Tensor [3,224,224]
```

最后：

```python
# [Contract] `imgs_t` is a list of Tensors of shape [3, input_size, input_size]; stack along dim = 0  
img_t = torch.vstack(imgs_t)
```

`[3, 224, 224] + [3, 224, 224]` 后沿 channel 维堆叠成 `[6, 224, 224]`

回到 `preprocess` 后经过 batch stack 得到 `[1, 6, 224, 224]`

```python
# Apply `self.img_transform` to each image (will return list of torch.Tensors); stack into "batched" Tensor  
pixel_values = torch.stack([self.apply_transform(img.convert("RGB")) for img in images])
```

这里正好对应 `PrismaticVisionBackbone.forward` 中的：

```python
# Split `pixel_values :: [bsz, 2 * 3, resolution, resolution]` =>> featurize =>> channel stack  
        img, img_fused = torch.split(pixel_values, [3, 3], dim=1)  # 先将拼接的图像信息拆开，得到 DINO input [1,3,224,224] 和 SigLIP input [1,3,224,224]
```

回顾上面，完成的图像数据链路图如下：

```
LIBERO RGB
     │
     ↓
PIL Image
     │
     ↓
PrismaticImageProcessor
     │
     ├── DINO preprocessing
     │      ↓
     │   [3,224,224]
     │
     └── SigLIP preprocessing
            ↓
         [3,224,224]
             │
             ↓ vstack
      pixel_values
      [1,6,224,224]
             │
════════════ MODEL ════════════
             │
             ↓ split
        ┌────┴────┐
        ↓         ↓
      DINO      SigLIP
        ↓         ↓
  [256,1024] [256,1152]
        └────┬────┘
             ↓
        [256,2176]
             ↓
         Projector
             ↓
        [256,4096]
```

于此同时文本处理如下：

```python
task.language
     ↓
prompt template
     ↓
LlamaTokenizer
     ↓
input_ids [1,T]
attention_mask [1,T]
     │
════════════ MODEL ════════════
     │
Embedding Table
     ↓
[T,4096]
```

