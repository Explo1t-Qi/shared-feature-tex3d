# Openpi

研究路线图：

```
Phase 0
openpi 仓库地图
       ↓
Phase 1
LIBERO → policy.infer() → actions
       ↓
Phase 1.5
image/state/language preprocessing
       ↓
Phase 2.1
VLM / visual representation
       ↓
Phase 2.2
Action Expert
       ↓
Phase 2.3
Flow Matching
       ↓
Phase 3
Representation audit 接口
       ↓
Phase 4
OpenVLA ↔ π₀ 对照
```

## Phase 0

```
┌─────────────────────────────────────┐
│ Layer 4：LIBERO Environment         │
│                                     │
│ examples/libero/main.py             │
│                                     │
│ RGB + wrist RGB + state + language  │
└──────────────────┬──────────────────┘
                   │ websocket
                   ↓
┌─────────────────────────────────────┐
│ Layer 3：Policy / Adapter           │
│                                     │
│ policies/libero_policy.py           │
│ policies/policy.py                  │
│ transforms.py                       │
│                                     │
│ Environment format                  │
│        ↓                            │
│ Model Observation                   │
└──────────────────┬──────────────────┘
                   │
                   ↓ sample_actions()
┌─────────────────────────────────────┐
│ Layer 2：π₀                         │
│                                     │
│ models/pi0.py              JAX      │
│ models_pytorch/pi0_pytorch.py Torch │
│                                     │
│ VLM + Action Expert + Flow Matching │
└──────────────────┬──────────────────┘
                   │
                   ↓
┌─────────────────────────────────────┐
│ Layer 1：Foundation Components      │
│                                     │
│ SigLIP                              │
│ Gemma / PaliGemma                   │
│ Action Expert                       │
└─────────────────────────────────────┘
```

## Phase 1

### client

首先按照 libero 数据集从数据输入开始，到模型输出动作：

```python
# examples/libero/main.py

def eval_libero(args: Args) -> None:  
    # Set random seed  
    np.random.seed(args.seed)  
  
    # Initialize LIBERO task suite  
    benchmark_dict = benchmark.get_benchmark_dict()  
    task_suite = benchmark_dict[args.task_suite_name]()  
    num_tasks_in_suite = task_suite.n_tasks  
    logging.info(f"Task suite: {args.task_suite_name}")  
  
    ...
	
	# 首先通过 Websocket 与模型建立通讯关系
    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    
    ...
    
    # 获取模型需要的图片数据，分别是 agentview_image 和 robot0_eye_in_hand_image。进行简单的预处理：翻转，填充，类型转换后准备送入模型
    # Get preprocessed image  
	# IMPORTANT: rotate 180 degrees to match train preprocessing
    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])  
	wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])  
	img = image_tools.convert_to_uint8(  
	    image_tools.resize_with_pad(img, args.resize_size, args.resize_size)  
	)  
	wrist_img = image_tools.convert_to_uint8(  
	    image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size)  
	)  
	  
	# Save preprocessed image for replay video  
	replay_images.append(img)  
	  
	if not action_plan:  
	    # Finished executing previous action chunk -- compute new chunk  
	    # Prepare observations dict    
	    # 准备数据时需要将上面 mujoco 使用键转换成 openpi 约定的输入数据类型
	    element = {  
	        "observation/image": img,  
	        "observation/wrist_image": wrist_img,  
	        "observation/state": np.concatenate(  
	            (  
	                obs["robot0_eef_pos"],  
	                _quat2axisangle(obs["robot0_eef_quat"]),  
	                obs["robot0_gripper_qpos"],  
	            )  
	        ),  
	        "prompt": str(task_description),  
	    }  
	  
	    # Query model to get action  
	    action_chunk = client.infer(element)["actions"]  # 向模型传入数据，最后通过 actions 这个键获取到真正的动作
	    assert (  
	        len(action_chunk) >= args.replan_steps  
	    ), f"We want to replan every {args.replan_steps} steps, but policy only predicts {len(action_chunk)} steps."    action_plan.extend(action_chunk[: args.replan_steps])  
	  
	action = action_plan.popleft()  
	  
	# Execute action in environment  
	obs, reward, done, info = env.step(action.tolist())  
	if done:  
	    task_successes += 1  
	    total_successes += 1  
	    break  
	t += 1
    
```

### server

上面是客户端，想分析为什么像上面那样传参，比如为什么 `element` 的字典需要那样构造，为什么同步 `actions` 键获取动作。还需要看服务端，根据 `README.md` 服务端启动 `scripts/serve_policy.py`:

```python

# Default checkpoints that should be used for each environment.  
DEFAULT_CHECKPOINT: dict[EnvMode, Checkpoint] = {  
    ...
    
    EnvMode.LIBERO: Checkpoint(  
        config="pi05_libero",  
        dir="gs://openpi-assets/checkpoints/pi05_libero",  
    ),  
}

def create_policy(args: Args) -> _policy.Policy:  
    """Create a policy from the given arguments."""  
    match args.policy:  
        case Checkpoint():  
            return _policy_config.create_trained_policy(  
                _config.get_config(args.policy.config), args.policy.dir, default_prompt=args.default_prompt  
            )  
        case Default():  
            return create_default_policy(args.env, default_prompt=args.default_prompt)  
  
  
def main(args: Args) -> None:  
    policy = create_policy(args)  
    policy_metadata = policy.metadata  
  
    # Record the policy's behavior.  
    if args.record:  
        policy = _policy.PolicyRecorder(policy, "policy_records")  
  
    hostname = socket.gethostname()  
    local_ip = socket.gethostbyname(hostname)  
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)  
	
	# 可以看到这里 server 接收了上面创建的 policy，这里接收的 policy 所含有的方法就是支持客户端调用的方法
    server = websocket_policy_server.WebsocketPolicyServer(  
        policy=policy,  
        host="0.0.0.0",  
        port=args.port,  
        metadata=policy_metadata,  
    )  
    server.serve_forever()
```

可以看到 `create_policy` 方法返回的类型是 `Policy`，而 `Policy` 实现的是 `BasePolicy` 这个抽象类，根据实现关系也可以看到这里的 `infer` 方法就是接口：

```python
class BasePolicy(abc.ABC):  
    @abc.abstractmethod  
    def infer(self, obs: Dict) -> Dict:  
        """Infer actions from observations."""  
  
    def reset(self) -> None:  
        """Reset the policy to its initial state."""  
        pass
```

下面具体来看 `infer` 方法的流程：

```python
class Policy(BasePolicy):  
    ...
  
    @override  
    def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:  # type: ignore[misc]  
        # Make a copy since transformations may modify the inputs in place.        
        inputs = jax.tree.map(lambda x: x, obs)  
        
        # 首先对输入数据做处理
        inputs = self._input_transform(inputs)  
        ...
		
		# 根据数据获取模型观察到的信息
        observation = _model.Observation.from_dict(inputs)  
        ...
        
        # 调用 sample_action 方法采样得到动作输出
        outputs = {  
            "state": inputs["state"],  
            "actions": self._sample_actions(sample_rng_or_pytorch_device, observation, **sample_kwargs),  
        }  
        ...
		
		# 最后对输出做变换
        outputs = self._output_transform(outputs)  
        ...
        
        return outputs  
  
    @property  
    def metadata(self) -> dict[str, Any]:  
        return self._metadata
```

至此最基础的客户端服务端通信数据流已经清晰：

```
════════════ LIBERO CLIENT ════════════

examples/libero/main.py

obs
 ↓
element
 ↓
client.infer(element)

════════════ WebSocket ════════════════

             ↓

════════════ MODEL SERVER ═════════════

Policy.infer(element)
 ↓
model.sample_actions(...)
```


接下来就根据服务端数据流来持续分析：

```
raw LIBERO dict
        │
        ▼
_input_transform
        │
        ▼
model-format dict
        │
        ▼
Observation.from_dict()
        │
        ▼
Pi0.sample_actions()
        │
        ▼
action chunk
        │
        ▼
_output_transform
        │
        ▼
LIBERO actions
```

### Phase 1.5

#### \_input\_transform

首先来看 `_input_transform` 到底对输入数据做了什么变换，以及为什么要接收这种构造的数据，这里开始追踪调用流：

```python
class Policy(BasePolicy):  
    def __init__(  
        self,  
        model: _model.BaseModel,  
        *,  
        rng: at.KeyArrayLike | None = None,  
        transforms: Sequence[_transforms.DataTransformFn] = (),  
        output_transforms: Sequence[_transforms.DataTransformFn] = (),  
        sample_kwargs: dict[str, Any] | None = None,  
        metadata: dict[str, Any] | None = None,  
        pytorch_device: str = "cpu",  
        is_pytorch: bool = False,  
    ):  
        """Initialize the Policy.  
  
        Args:            
        transforms: Input data transformations to apply before inference.
        """        
        ...
		self._model = model  
		self._input_transform = _transforms.compose(transforms)  
		self._output_transform = _transforms.compose(output_transforms)  
		self._sample_kwargs = sample_kwargs or {}  
		self._metadata = metadata or {}  
		self._is_pytorch_model = is_pytorch  
		self._pytorch_device = pytorch_device  
		  
		if self._is_pytorch_model:  
		    self._model = self._model.to(pytorch_device)  
		    self._model.eval()  
		    self._sample_actions = model.sample_actions  
		else:  
		    # JAX model setup  
		    self._sample_actions = nnx_utils.module_jit(model.sample_actions)  
		    self._rng = rng or jax.random.key(0)
        
```

所以需要去分析创建时的参数调用，回到 `scripts/serve_policy.py`，这里调用方是 `create_trained_policy` ，还需要看懂参数是如何传递的，因为我们研究的是 `pi0 + libero` 数据集，所以 `args.policy.config` 理论上应该是 `pi0_libero`，但是这里并没有提供，所以就按照 `pi05_libero` 来看，很明显获取的就是一个代表配置的字符串：

```python

# Default checkpoints that should be used for each environment.  
DEFAULT_CHECKPOINT: dict[EnvMode, Checkpoint] = {  
    ...
    
    EnvMode.LIBERO: Checkpoint(  
        config="pi05_libero",  
        dir="gs://openpi-assets/checkpoints/pi05_libero",  
    ),  
}
```

```python
def create_policy(args: Args) -> _policy.Policy:  
    """Create a policy from the given arguments."""  
    match args.policy:  
        case Checkpoint():  
            return _policy_config.create_trained_policy(  
                _config.get_config(args.policy.config), args.policy.dir, default_prompt=args.default_prompt  
            )  
        case Default():  
            return create_default_policy(args.env, default_prompt=args.default_prompt)
```

知晓 `_config.get_config` 接收的参数后，我们来看这里的是如何获取对应的配置的，这里假设是 `pi0_libero`：

```python
# src/openpi/training/config.py
# Use `get_config` if you need to get a config by name in your code.  
_CONFIGS = [  
    ...
    #  
    # Fine-tuning Libero configs.    
    #    
    # These train configs define the hyperparameters for fine-tuning the base model on your own dataset.    
    # They are used to define key elements like the dataset you are training on, the base checkpoint you    
    # are using, and other hyperparameters like how many training steps to run or what learning rate to use.    
    # For your own dataset, you can copy this class and modify the dataset name, and data transforms based on    
    # the comments below.    
    TrainConfig(  
        # Change the name to reflect your model and dataset.  
        name="pi0_libero",  
        # Here you define the model config -- In this example we use pi0 as the model  
        # architecture and perform *full* finetuning. in the examples below we show how to modify        
        # this to perform *low-memory* (LORA) finetuning and use pi0-FAST as an alternative architecture.        
        model=pi0_config.Pi0Config(),  
        # Here you define the dataset you are training on. In this example we use the Libero  
        # dataset. For your own dataset, you can change the repo_id to point to your dataset.       
        # Also modify the DataConfig to use the new config you made for your dataset above.        
        data=LeRobotLiberoDataConfig(  
            repo_id="physical-intelligence/libero",  
            base_config=DataConfig(  
                # This flag determines whether we load the prompt (i.e. the task instruction) from the  
                # ``task`` field in the LeRobot dataset. If set to True, the prompt will show up in                
                # a field called ``prompt`` in the input dict. The recommended setting is True.                
                prompt_from_task=True,  
            ),  
            extra_delta_transform=True,  
        ),  
        # Here you define which pre-trained checkpoint you want to load to initialize the model.  
        # This should match the model config you chose above -- i.e. in this case we use the pi0 base model.        
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),  
        # Below you can define other hyperparameters like the learning rate, number of training steps, etc.  
        # Check the base TrainConfig class for a full list of available hyperparameters.        
        num_train_steps=30_000,  
    )
...

_CONFIGS_DICT = {config.name: config for config in _CONFIGS}
...

def get_config(config_name: str) -> TrainConfig:  
    """Get a config by name."""  
    if config_name not in _CONFIGS_DICT:  
        closest = difflib.get_close_matches(config_name, _CONFIGS_DICT.keys(), n=1, cutoff=0.0)  
        closest_str = f" Did you mean '{closest[0]}'? " if closest else ""  
        raise ValueError(f"Config '{config_name}' not found.{closest_str}")  
  
    return _CONFIGS_DICT[config_name]
```

继续深入来看 `TrainConfig` 中的 `LeRobotLiberoDataConfig` 就可以找到输入数据的原型：

```python
# src/openpi/training/config.py
@dataclasses.dataclass(frozen=True)  
class LeRobotLiberoDataConfig(DataConfigFactory):  
    """  
    This config is used to configure transforms that are applied at various parts of the data pipeline.    
    For your own dataset, you can copy this class and modify the transforms to match your dataset based on the    
    comments below.    
    """  
    extra_delta_transform: bool = False  
  
    @override  
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:  
        # The repack transform is *only* applied to the data coming from the dataset,  
        # and *not* during inference. We can use it to make inputs from the dataset look        
        # as close as possible to those coming from the inference environment (e.g. match the keys).        
        # Below, we match the keys in the dataset (which we defined in the data conversion script) to        
        # the keys we use in our inference pipeline (defined in the inference script for libero).        
        # For your own dataset, first figure out what keys your environment passes to the policy server        
        # and then modify the mappings below so your dataset's keys get matched to those target keys.        
        # The repack transform simply remaps key names here.    
        # 在 create_trained_policy 出现使用    
        repack_transform = _transforms.Group(  
            inputs=[  
                _transforms.RepackTransform(  
                    {  
                        "observation/image": "image",  
                        "observation/wrist_image": "wrist_image",  
                        "observation/state": "state",  
                        "actions": "actions",  
                        "prompt": "prompt",  
                    }  
                )  
            ]  
        )  
  
        # The data transforms are applied to the data coming from the dataset *and* during inference.  
        # Below, we define the transforms for data going into the model (``inputs``) and the transforms        
        # for data coming out of the model (``outputs``) (the latter is only used during inference).        
        # We defined these transforms in `libero_policy.py`. You can check the detailed comments there for        
        # how to modify the transforms to match your dataset. Once you created your own transforms, you can        
        # replace the transforms below with your own. 
        # 在 create_trained_policy 出现使用 
        data_transforms = _transforms.Group(  
            inputs=[libero_policy.LiberoInputs(model_type=model_config.model_type)],  
            outputs=[libero_policy.LiberoOutputs()],  
        )  
  
        ...
        
        # Model transforms include things like tokenizing the prompt and action targets  
		# You do not need to change anything here for your own dataset.  
		model_transforms = ModelTransformFactory()(model_config)  
		  
		# We return all data transforms for training and inference. No need to change anything here.  
		return dataclasses.replace(  
		    self.create_base_config(assets_dirs, model_config),  
		    repack_transforms=repack_transform,  
		    data_transforms=data_transforms,  
		    model_transforms=model_transforms,  
		)
```

`model_transforms = ModelTransformFactory()(model_config)` 中对应的 pi0 模型的变换如下：

```python
# src/openpi/training/config.py
def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:  
    match model_config.model_type:  
        case _model.ModelType.PI0:  
            return _transforms.Group(  
                inputs=[  
                    _transforms.InjectDefaultPrompt(self.default_prompt),  
                    _transforms.ResizeImages(224, 224),  
                    _transforms.TokenizePrompt(  
                        _tokenizer.PaligemmaTokenizer(model_config.max_token_len),  
                    ),  
                    _transforms.PadStatesAndActions(model_config.action_dim),  
                ],  
            )
```

可以看到这里的 `data_transforms` 及其注释说明了接收的输入数据和输出数据的具体格式。看到这里就清楚了为什么在 `client` 调用 `infer` 方法时接收的参数类型是:

```python
		element = {
	        "observation/image": img,  
	        "observation/wrist_image": wrist_img,  
	        "observation/state": np.concatenate(  
	            (  
	                obs["robot0_eef_pos"],  
	                _quat2axisangle(obs["robot0_eef_quat"]),  
	                obs["robot0_gripper_qpos"],  
	            )  
	        ),  
	        "prompt": str(task_description),  
	    } 
```

这里存在的是 `mujoco -> args -> model input` 三种类型的数据的转换：

```python
# src/openpi/policies/libero_policy.py

...

@dataclasses.dataclass(frozen=True)  
class LiberoInputs(transforms.DataTransformFn):  
    """  
    This class is used to convert inputs to the model to the expected format. It is used for both training and inference.  
    For your own dataset, you can copy this class and modify the keys based on the comments below to pipe    
    the correct elements of your dataset into the model.    
    """  
    # Determines which model will be used.  
    # Do not change this for your own dataset.    
    model_type: _model.ModelType  
  
    def __call__(self, data: dict) -> dict:  
        # Possibly need to parse images to uint8 (H,W,C) since LeRobot automatically  
        # stores as float32 (C,H,W), gets skipped for policy inference.        
        # Keep this for your own dataset, but if your dataset stores the images        
        # in a different key than "observation/image" or "observation/wrist_image",        
        # you should change it below.        
        # Pi0 models support three image inputs at the moment: one third-person view,        
        # and two wrist views (left and right). If your dataset does not have a particular type        
        # of image, e.g. wrist images, you can comment it out here and replace it with zeros like we do for the        
        # right wrist image below.        
        base_image = _parse_image(data["observation/image"])  
        wrist_image = _parse_image(data["observation/wrist_image"])  
  
        # Create inputs dict. Do not change the keys in the dict below.  
        inputs = {  
            "state": data["observation/state"],  
            # 这里就可以确认 pi0 的接口最多只有 3 个 camera slot，而 libero 只有 2 个，即 base_0_rgb <- agentview, left_wrist_0_rgb <- eye-in-hand 所以最后一个直接使用零填充
            "image": {  
                "base_0_rgb": base_image,  
                "left_wrist_0_rgb": wrist_image,  
                # Pad any non-existent images with zero-arrays of the appropriate shape.  
                "right_wrist_0_rgb": np.zeros_like(base_image),  
            },  
            "image_mask": {  
                "base_0_rgb": np.True_,  
                "left_wrist_0_rgb": np.True_,  
                # We only mask padding images for pi0 model, not pi0-FAST. Do not change this for your own dataset.  
                "right_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,  
            },  
        }  
  
        # Pad actions to the model action dimension. Keep this for your own dataset.  
        # Actions are only available during training.        
        if "actions" in data:  
            inputs["actions"] = data["actions"]  
  
        # Pass the prompt (aka language instruction) to the model.  
        # Keep this for your own dataset (but modify the key if the instruction is not        
        # stored in "prompt"; the output dict always needs to have the key "prompt").        
        if "prompt" in data:  
            inputs["prompt"] = data["prompt"]  
  
        return inputs  
  
# 同样从这里就可以看出为什么最后是获取输出使用 actions 这个键
@dataclasses.dataclass(frozen=True)  
class LiberoOutputs(transforms.DataTransformFn):  
    """  
    This class is used to convert outputs from the model back the the dataset specific format. It is    
    used for inference only.  
    For your own dataset, you can copy this class and modify the action dimension based on the comments below.    
    """  
    def __call__(self, data: dict) -> dict:  
        # Only return the first N actions -- since we padded actions above to fit the model action  
        # dimension, we need to now parse out the correct number of actions in the return dict.        
        # For Libero, we only return the first 7 actions (since the rest is padding).        
        # For your own dataset, replace `7` with the action dimension of your dataset.        
        return {"actions": np.asarray(data["actions"][..., :7])}
```

现在从 `create_policy -> get_config -> LeRobotLiberoDataConfig -> LiberoInputs` 的调用链回归，继续看 `create_policy` 调用的 `create_trained_policy` 方法。

`Policy` 中的 `_input_transform` 调用的就是 `create_trained_policy` 这里的 `transforms`:

```python
self._input_transform = _transforms.compose(transforms) 
```

这里就明确了传入 `create_trained_policy` 中返回的 `transforms` 具体是什么，`transforms` 中还有 `repack_transforms`，它在 `LeRobotLiberoDataConfig.create` 中也出现了：
``
```python
def create_trained_policy(  
    train_config: _config.TrainConfig,  
    checkpoint_dir: pathlib.Path | str,  
    *,  
    repack_transforms: transforms.Group | None = None,  
    sample_kwargs: dict[str, Any] | None = None,  
    default_prompt: str | None = None,  
    norm_stats: dict[str, transforms.NormStats] | None = None,  
    pytorch_device: str | None = None,  
) -> _policy.Policy:  
    """
    Create a policy from a trained checkpoint.  
    """    
    repack_transforms = repack_transforms or transforms.Group()  
    checkpoint_dir = download.maybe_download(str(checkpoint_dir))  
  
    # Check if this is a PyTorch model by looking for model.safetensors  
    weight_path = os.path.join(checkpoint_dir, "model.safetensors")  
    is_pytorch = os.path.exists(weight_path)  
  
    logging.info("Loading model...")  
    if is_pytorch:  
        model = train_config.model.load_pytorch(train_config, weight_path)  
        model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")  
    else:  
        model = train_config.model.load(_model.restore_params(checkpoint_dir / "params", dtype=jnp.bfloat16))  
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)  
    if norm_stats is None:  
        # We are loading the norm stats from the checkpoint instead of the config assets dir to make sure  
        # that the policy is using the same normalization stats as the original training process.        if data_config.asset_id is None:  
            raise ValueError("Asset id is required to load norm stats.")  
        norm_stats = _checkpoints.load_norm_stats(checkpoint_dir / "assets", data_config.asset_id)  
  
    # Determine the device to use for PyTorch models  
    if is_pytorch and pytorch_device is None:  
        try:  
            import torch  
  
            pytorch_device = "cuda" if torch.cuda.is_available() else "cpu"  
        except ImportError:  
            pytorch_device = "cpu"  
  
    return _policy.Policy(  
        model,  
        transforms=[  
            *repack_transforms.inputs,  # 对应 LeRobotLiberoDataConfig 中的 inputs
            transforms.InjectDefaultPrompt(default_prompt),  
            *data_config.data_transforms.inputs,  # 对应 LeRobotLiberoDataConfig.create 中的 data_transforms 
            transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),  
            *data_config.model_transforms.inputs,  
        ],  
        output_transforms=[  
            *data_config.model_transforms.outputs,  
            transforms.Unnormalize(norm_stats, use_quantiles=data_config.use_quantile_norm),  
            *data_config.data_transforms.outputs,  
            *repack_transforms.outputs,  
        ],  
        sample_kwargs=sample_kwargs,  
        metadata=train_config.policy_metadata,  
        is_pytorch=is_pytorch,  
        pytorch_device=pytorch_device if is_pytorch else None,  
    )
```

至此对输入做的变换已经清晰，流程如下：

```
RepackTransform -> InjectDefaultPrompt -> libero_policy.LiberoInputs -> Normalize -> model_transforms

model_transforms: InjectDefaultPrompt -> ResizeImages (224 * 224) -> TokenizePrompt(PaligemmaTokenizer) -> PadStatesAndActions
```

#### Observation.from_dict

```python
@at.typecheck  
@struct.dataclass  
class Observation(Generic[ArrayT]):  
    """
    Holds observations, i.e., inputs to the model.  
  
    See `Observation.from_dict` to see the expected dictionary form. This is the format    
    that should be produced by the data transforms.    
    """  
    # Images, in [-1, 1] float32.  
    images: dict[str, at.Float[ArrayT, "*b h w c"]]  
    # Image masks, with same keys as images.  
    image_masks: dict[str, at.Bool[ArrayT, "*b"]]  
    # Low-dimensional robot state.  
    state: at.Float[ArrayT, "*b s"]  
  
    # Tokenized prompt.  
    tokenized_prompt: at.Int[ArrayT, "*b l"] | None = None  
    # Tokenized prompt mask.  
    tokenized_prompt_mask: at.Bool[ArrayT, "*b l"] | None = None  
  
    # pi0-fast model specific fields.  
  
    # Token auto-regressive mask (for FAST autoregressive model).    token_ar_mask: at.Int[ArrayT, "*b l"] | None = None  
    # Token loss mask (for FAST autoregressive model).  
    token_loss_mask: at.Bool[ArrayT, "*b l"] | None = None  
  
    @classmethod  
    def from_dict(cls, data: at.PyTree[ArrayT]) -> "Observation[ArrayT]":  
        """This method defines the mapping between unstructured data (i.e., nested dict) to the structured Observation format."""  
        # Ensure that tokenized_prompt and tokenized_prompt_mask are provided together.  
        if ("tokenized_prompt" in data) != ("tokenized_prompt_mask" in data):  
            raise ValueError("tokenized_prompt and tokenized_prompt_mask must be provided together.")  
        # If images are uint8, convert them to [-1, 1] float32.  
        for key in data["image"]:  
            if data["image"][key].dtype == np.uint8:  
                data["image"][key] = data["image"][key].astype(np.float32) / 255.0 * 2.0 - 1.0  # 对 uint8 的图像数据做标准化处理
            elif hasattr(data["image"][key], "dtype") and data["image"][key].dtype == torch.uint8:  
                data["image"][key] = data["image"][key].to(torch.float32).permute(0, 3, 1, 2) / 255.0 * 2.0 - 1.0  
        return cls(  
            images=data["image"],  
            image_masks=data["image_mask"],  
            state=data["state"],  
            tokenized_prompt=data.get("tokenized_prompt"),  
            tokenized_prompt_mask=data.get("tokenized_prompt_mask"),  
            token_ar_mask=data.get("token_ar_mask"),  
            token_loss_mask=data.get("token_loss_mask"),  
        )  
  
    def to_dict(self) -> at.PyTree[ArrayT]:  
        """Convert the Observation to a nested dict."""  
        result = dataclasses.asdict(self)  
        result["image"] = result.pop("images")  
        result["image_mask"] = result.pop("image_masks")  
        return result
```



```
LIBERO
  │
  ├── agentview_image
  │
  └── wrist_image
          │
          ▼
rotate 180°
          │
          ▼
resize_with_pad(224,224)
          │
          ▼
uint8 [0,255]
          │
          ▼
LiberoInputs
          │
          ├── base_0_rgb
          ├── left_wrist_0_rgb
          └── right_wrist_0_rgb = zeros
          │
          ▼
Model transforms
          │
          ├── prompt tokenization
          └── ...
          │
          ▼
Observation.from_dict()
          │
          ▼
float32 [-1,1]

```

#### \_sample\_action

这里需要看模型真正实现的该方法，之前在 `TrainConfig` 时看到这里真正调用的模型配置 `Pi0Config`:

```python
# src/openpi/training/config.py
TrainConfig(  
    name="pi0_libero_low_mem_finetune",  
    # Here is an example of loading a pi0 model for LoRA fine-tuning.  
    model=pi0_config.Pi0Config(paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"),  
    data=LeRobotLiberoDataConfig(  
        repo_id="physical-intelligence/libero",  
        base_config=DataConfig(prompt_from_task=True),  
        extra_delta_transform=True,  
    ),  
    weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),  
    num_train_steps=30_000,  
    # The freeze filter defines which parameters should be frozen during training.  
    # We have a convenience function in the model config that returns the default freeze filter    # for the given model config for LoRA finetuning. Just make sure it matches the model config    # you chose above.    
    freeze_filter=pi0_config.Pi0Config(  
        paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"  
    ).get_freeze_filter(),  
    # Turn off EMA for LoRA finetuning.  
    ema_decay=None,  
),
```

可以看到该配置有 `create` 方法创建模型：

```python
@dataclasses.dataclass(frozen=True)  
class Pi0Config(_model.BaseModelConfig):  
    dtype: str = "bfloat16"  
    paligemma_variant: _gemma.Variant = "gemma_2b"  
    action_expert_variant: _gemma.Variant = "gemma_300m"  
  
    # Set the model specific defaults.  
    action_dim: int = 32  
    action_horizon: int = 50  
    max_token_len: int = None  # type: ignore  
    # Pi05 has two differences from Pi0:    # - the state input is part of the discrete language tokens rather than a continuous input that is part of the suffix    # - the action expert uses adaRMSNorm to inject the flow matching timestep    pi05: bool = False  
    # This config option is not used directly by the model, but it is read by the ModelTransformFactory.  
    discrete_state_input: bool = None  # type: ignore  
  
    pytorch_compile_mode: str | None = "max-autotune"  
  
	...
  
    @override  
    def create(self, rng: at.KeyArrayLike) -> "Pi0":  
        from openpi.models.pi0 import Pi0  
  
        return Pi0(self, rngs=nnx.Rngs(rng))  
  
    ...
```

据此找到模型真正的位置，以及真正调用的 `Pi0.sample_action`：

```python
# src/openpi/models/pi0.py
class Pi0(_model.BaseModel):  
    def __init__(self, config: pi0_config.Pi0Config, rngs: nnx.Rngs):  
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)  
        self.pi05 = config.pi05  
        paligemma_config = _gemma.get_config(config.paligemma_variant)  
        action_expert_config = _gemma.get_config(config.action_expert_variant)  
        # TODO: rewrite gemma in NNX. For now, use bridge.  
        
        # 语言处理使用 gemma
        llm = nnx_bridge.ToNNX(  
            _gemma.Module(  
                configs=[paligemma_config, action_expert_config],  
                embed_dtype=config.dtype,  
                adarms=config.pi05,  
            )  
        )  
        llm.lazy_init(rngs=rngs, method="init", use_adarms=[False, True] if config.pi05 else [False, False])  
        
        # 图像处理使用 siglip
        img = nnx_bridge.ToNNX(  
            _siglip.Module(  
                num_classes=paligemma_config.width,  
                variant="So400m/14",  
                pool_type="none",  
                scan=True,  
                dtype_mm=config.dtype,  
            )  
        )  
        img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)  
        
        # 组装出 PaliGemma VLM 模型
        self.PaliGemma = nnx.Dict(llm=llm, img=img)  
        
        self.action_in_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)  
        if config.pi05:  
            self.time_mlp_in = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)  
            self.time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)  
        else:  
            self.state_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)  
            self.action_time_mlp_in = nnx.Linear(2 * action_expert_config.width, action_expert_config.width, rngs=rngs)  
            self.action_time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)  
        self.action_out_proj = nnx.Linear(action_expert_config.width, config.action_dim, rngs=rngs)  
  
        # This attribute gets automatically set by model.train() and model.eval().  
        self.deterministic = True  
  
    @at.typecheck  
    def embed_prefix(  
        self, obs: _model.Observation  
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:  
        ...
  
    @at.typecheck  
    def embed_suffix(  
        self, obs: _model.Observation, noisy_actions: _model.Actions, timestep: at.Float[at.Array, " b"]  
    ) -> tuple[  
        at.Float[at.Array, "b s emb"],  
        at.Bool[at.Array, "b s"],  
        at.Bool[at.Array, " s"],  
        at.Float[at.Array, "b emb"] | None,  
    ]:  
        ... 
  
    @override  
    def compute_loss(  
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False  
    ) -> at.Float[at.Array, "*b ah"]:  
        ...
  
    @override  
    def sample_actions(  
        self,  
        rng: at.KeyArrayLike,  
        observation: _model.Observation,  
        *,  
        num_steps: int | at.Int[at.Array, ""] = 10,  
        noise: at.Float[at.Array, "b ah ad"] | None = None,  
    ) -> _model.Actions:  
        observation = _model.preprocess_observation(None, observation, train=False)  
        # note that we use the convention more common in diffusion literature, where t=1 is noise and t=0 is the target  
        # distribution. yes, this is the opposite of the pi0 paper, and I'm sorry.        dt = -1.0 / num_steps  
        batch_size = observation.state.shape[0]  
        if noise is None:  
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))  
  
        # first fill KV cache with a forward pass of the prefix  
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)  
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)  
        positions = jnp.cumsum(prefix_mask, axis=1) - 1  
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)  
  
        def step(carry):  
            x_t, time = carry  
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(  
                observation, x_t, jnp.broadcast_to(time, batch_size)  
            )  
            # `suffix_attn_mask` is shape (b, suffix_len, suffix_len) indicating how the suffix tokens can attend to each  
            # other            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)  
            # `prefix_attn_mask` is shape (b, suffix_len, prefix_len) indicating how the suffix tokens can attend to the  
            # prefix tokens            prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])  
            # `combined_mask` is shape (b, suffix_len, prefix_len + suffix_len) indicating how the suffix tokens (which  
            # generate the queries) can attend to the full prefix + suffix sequence (which generates the keys and values)            
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)  
            assert full_attn_mask.shape == (  
                batch_size,  
                suffix_tokens.shape[1],  
                prefix_tokens.shape[1] + suffix_tokens.shape[1],  
            )  
            # `positions` is shape (b, suffix_len) indicating the positions of the suffix tokens  
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1  
  
            (prefix_out, suffix_out), _ = self.PaliGemma.llm(  
                [None, suffix_tokens],  
                mask=full_attn_mask,  
                positions=positions,  
                kv_cache=kv_cache,  
                adarms_cond=[None, adarms_cond],  
            )  
            assert prefix_out is None  
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])  
  
            return x_t + dt * v_t, time + dt  
  
        def cond(carry):  
            x_t, time = carry  
            # robust to floating-point error  
            return time >= -dt / 2  
  
        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))  
        return x_0
```

在`sample_actions`中第一个调用的本类中的方法是`embed_prefix`，这里第一次使用 `siglip` 获取到 `image_tokens`：

```python
        inputs = {  
            "state": data["observation/state"],  
            # 这里就可以确认 pi0 的接口最多只有 3 个 camera slot，而 libero 只有 2 个，即 base_0_rgb <- agentview, left_wrist_0_rgb <- eye-in-hand 所以最后一个直接使用零填充
            "image": {  
                "base_0_rgb": base_image,  
                "left_wrist_0_rgb": wrist_image,  
                # Pad any non-existent images with zero-arrays of the appropriate shape.  
                "right_wrist_0_rgb": np.zeros_like(base_image),  
            },  
            "image_mask": {  
                "base_0_rgb": np.True_,  
                "left_wrist_0_rgb": np.True_,  
                # We only mask padding images for pi0 model, not pi0-FAST. Do not change this for your own dataset.  
                "right_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,  
            },  
        }  
```

根据数据类型可以看到 `for` 循环逐相机获取图像进行编码，然后 `llm` 编码 `prompt`，最后将编码得到的 `token` 合并：

```python
@at.typecheck  
def embed_prefix(  
    self, obs: _model.Observation  
) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:  
    input_mask = []  
    ar_mask = []  
    tokens = []  
    # embed images  
    for name in obs.images:  
        image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)  
  
        tokens.append(image_tokens)  
        input_mask.append(  
            einops.repeat(  
                obs.image_masks[name],  
                "b -> b s",  
                s=image_tokens.shape[1],  
            )  
        )  
        # image tokens attend to each other  
        ar_mask += [False] * image_tokens.shape[1]  
  
    # add language (aka tokenized inputs)  
    if obs.tokenized_prompt is not None:  
        tokenized_inputs = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")  
        tokens.append(tokenized_inputs)  
        input_mask.append(obs.tokenized_prompt_mask)  
        # full attention between image and language inputs  
        ar_mask += [False] * tokenized_inputs.shape[1]  
    tokens = jnp.concatenate(tokens, axis=1)  
    input_mask = jnp.concatenate(input_mask, axis=1)  
    ar_mask = jnp.array(ar_mask)  
    return tokens, input_mask, ar_mask
```

所以得到的 `prefix` 是：

```python
[base image tokens]
[left wrist image tokens]
[right wrist image tokens] # libero 数据集的话，这张图片对应的是 0 零填充的数据，编码后也会较为特殊，没有实际意义
[language tokens]
```

所以，回顾以上内容，需要掌握以下问题：


> [!NOTE]- LIBERO 给 π0 几张真实 camera image？
> 两张，分别是 agent_view 和 wrist

> [!NOTE]- 为什么 model observation 中却固定出现三张 image？
> Pi0 模型输入约定是 3 张，第三张使用 0 填充

> [!NOTE]- `client.infer()` 和 `Policy.infer()` 是不是同一个函数？
> 不是，一个是客户端发出查询请求的函数，一个是服务端催动模型推理的函数

> [!NOTE]- `LiberoInputs` 起什么作用？
> 定义模型输入格式

> [!NOTE]- `Observation.from_dict()` 对 image 又做了什么？
> 标准化图像数据，将 uint8 变成 float32[-1, 1]，并转成 Observation

> [!NOTE]- `Observation.from_dict()` 对 image 又做了什么？
> 标准化图像数据，将 uint8 变成 float32[-1, 1]，并转成 Observation

> [!NOTE]- 第一份真正的 π0 visual token 在哪一行产生？
> image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)  

> [!NOTE]- `embed_prefix()` 中 image、language、state 谁进入 prefix，谁没有进入？
> image + language 进入， state 没有进入
