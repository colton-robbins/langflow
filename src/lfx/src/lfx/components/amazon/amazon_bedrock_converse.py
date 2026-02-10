from langflow.field_typing import LanguageModel
from langflow.inputs.inputs import FloatInput, IntInput, MessageTextInput, SecretStrInput
from langflow.io import DictInput, DropdownInput

from lfx.base.models.aws_constants import AWS_REGIONS, AWS_MODEL_IDs
from lfx.base.models.model import LCModelComponent


class AmazonBedrockConverseComponent(LCModelComponent):
    display_name: str = "ANSWER_AWS"
    description: str = (
        "Generate text using Amazon Bedrock LLMs with the invoke_model API for improved performance."
    )
    icon = "Amazon"
    name = "AmazonBedrockConverseModel"
    beta = True

    def __init__(self, **kwargs):
        """Initialize the component and create caches for boto3 session and client."""
        super().__init__(**kwargs)
        self._cached_boto3_session = None
        self._cached_boto3_client = None
        self._cached_model = None
        self._cached_model_key = None


    inputs = [
        *LCModelComponent.get_base_inputs(),
        MessageTextInput(
            name="model_id",
            display_name="Model ID",
            value="anthropic.claude-3-5-sonnet-20241022-v2:0",
            info="Enter the model ID (e.g., 'anthropic.claude-3-5-sonnet-20241022-v2:0'). "
            "You can use any custom model name and include inference profile versions using the format 'model_name:profile_version' "
            "(e.g., 'anthropic.claude-3-5-sonnet-20241022-v2:1' for profile version 1). "
            f"Common models: {', '.join(AWS_MODEL_IDs[:5])}...",
        ),
        SecretStrInput(
            name="aws_access_key_id",
            display_name="AWS Access Key ID",
            info="The access key for your AWS account. "
            "Usually set in Python code as the environment variable 'AWS_ACCESS_KEY_ID'.",
            value="AWS_ACCESS_KEY_ID",
            required=True,
        ),
        SecretStrInput(
            name="aws_secret_access_key",
            display_name="AWS Secret Access Key",
            info="The secret key for your AWS account. "
            "Usually set in Python code as the environment variable 'AWS_SECRET_ACCESS_KEY'.",
            value="AWS_SECRET_ACCESS_KEY",
            required=True,
        ),
        SecretStrInput(
            name="aws_session_token",
            display_name="AWS Session Token",
            advanced=True,
            info="The session key for your AWS account. "
            "Only needed for temporary credentials. "
            "Usually set in Python code as the environment variable 'AWS_SESSION_TOKEN'.",
            load_from_db=False,
        ),
        SecretStrInput(
            name="credentials_profile_name",
            display_name="Credentials Profile Name",
            advanced=True,
            info="The name of the profile to use from your "
            "~/.aws/credentials file. "
            "If not provided, the default profile will be used.",
            load_from_db=False,
        ),
        DropdownInput(
            name="region_name",
            display_name="Region Name",
            value="us-east-1",
            options=AWS_REGIONS,
            info="The AWS region where your Bedrock resources are located.",
        ),
        MessageTextInput(
            name="endpoint_url",
            display_name="Endpoint URL",
            advanced=True,
            info="The URL of the Bedrock endpoint to use.",
        ),
        # Model-specific parameters for fine control
        DropdownInput(
            name="sampling_strategy",
            display_name="Sampling Strategy",
            options=["temperature", "top_p"],
            value="temperature",
            info="Some models (like Claude Opus 4.5) do not allow temperature and top_p together. Choose which one to use.",
            advanced=True,
        ),
        FloatInput(
            name="temperature",
            display_name="Temperature",
            value=0.7,
            info="Controls randomness in output. Higher values make output more random. Only used if Sampling Strategy is set to 'temperature'.",
            advanced=True,
        ),
        IntInput(
            name="max_tokens",
            display_name="Max Tokens",
            value=4096,
            info="Maximum number of tokens to generate.",
            advanced=True,
        ),
        FloatInput(
            name="top_p",
            display_name="Top P",
            value=0.9,
            info="Nucleus sampling parameter. Controls diversity of output. Only used if Sampling Strategy is set to 'top_p'.",
            advanced=True,
        ),
        IntInput(
            name="top_k",
            display_name="Top K",
            value=250,
            info="Limits the number of highest probability vocabulary tokens to consider. "
            "Note: Not all models support top_k. Use 'Additional Model Fields' for manual configuration if needed.",
            advanced=True,
        ),
        DictInput(
            name="additional_model_fields",
            display_name="Additional Model Fields",
            advanced=True,
            is_list=True,
            info="Additional model-specific parameters for fine-tuning behavior.",
        ),
    ]

    def build_model(self) -> LanguageModel:  # type: ignore[type-var]
        try:
            from langchain_aws import ChatBedrock
        except ImportError as e:
            msg = "langchain_aws is not installed. Please install it with `pip install langchain_aws`."
            raise ImportError(msg) from e
        try:
            import boto3
            from botocore.config import Config
        except ImportError as e:
            msg = "boto3 is not installed. Please install it with `pip install boto3`."
            raise ImportError(msg) from e
        
        # Cache boto3 session to avoid creating multiple sessions
        if self._cached_boto3_session is None:
            if self.aws_access_key_id or self.aws_secret_access_key:
                try:
                    self._cached_boto3_session = boto3.Session(
                        aws_access_key_id=self.aws_access_key_id,
                        aws_secret_access_key=self.aws_secret_access_key,
                        aws_session_token=self.aws_session_token,
                    )
                except Exception as e:
                    msg = "Could not create a boto3 session."
                    raise ValueError(msg) from e
            elif self.credentials_profile_name:
                self._cached_boto3_session = boto3.Session(profile_name=self.credentials_profile_name)
            else:
                self._cached_boto3_session = boto3.Session()
        
        session = self._cached_boto3_session

        # Optimize boto3 client configuration for connection reuse and performance
        boto_config = Config(
            max_pool_connections=50,
            connect_timeout=10,
            read_timeout=60,
            retries={
                "max_attempts": 3,
                "mode": "adaptive"
            }
        )

        client_params = {
            "config": boto_config
        }
        if self.endpoint_url:
            client_params["endpoint_url"] = self.endpoint_url
        if self.region_name:
            client_params["region_name"] = self.region_name

        # Cache boto3 client to avoid creating multiple clients
        if self._cached_boto3_client is None:
            self._cached_boto3_client = session.client("bedrock-runtime", **client_params)
        
        boto3_client = self._cached_boto3_client
        
        # Build model_kwargs from individual parameters
        # ChatBedrock expects parameters directly in model_kwargs using snake_case
        # It will handle conversion to camelCase and wrapping in inferenceConfig internally
        model_kwargs = {}
        
        # Add model parameters - only send one of temperature or top_p based on sampling_strategy
        sampling_strategy = getattr(self, "sampling_strategy", "temperature")
        
        if sampling_strategy == "top_p":
            if hasattr(self, "top_p") and self.top_p is not None:
                model_kwargs["top_p"] = self.top_p
        else:
            # Default to temperature if not specified or if sampling_strategy is "temperature"
            if hasattr(self, "temperature") and self.temperature is not None:
                model_kwargs["temperature"] = self.temperature
        
        if hasattr(self, "max_tokens") and self.max_tokens is not None:
            model_kwargs["max_tokens"] = self.max_tokens
        
        # Note: top_k may not be supported by all models or for streaming
        # Only add if explicitly provided and not None
        if hasattr(self, "top_k") and self.top_k is not None:
            # Some models don't support top_k, so we'll let ChatBedrock handle validation
            model_kwargs["top_k"] = self.top_k

        # Handle additional model fields
        # ChatBedrock expects parameters at top level of model_kwargs in snake_case
        if hasattr(self, "additional_model_fields") and self.additional_model_fields:
            for field in self.additional_model_fields:
                if isinstance(field, dict):
                    # If field contains inferenceConfig, extract its contents
                    if "inferenceConfig" in field and isinstance(field["inferenceConfig"], dict):
                        # Merge inferenceConfig contents directly
                        # ChatBedrock will handle any format conversion needed
                        for key, value in field["inferenceConfig"].items():
                            model_kwargs[key] = value
                    else:
                        # Merge other fields directly (excluding inferenceConfig wrapper)
                        for key, value in field.items():
                            if key != "inferenceConfig":
                                model_kwargs[key] = value

        # Create a cache key based on model parameters to avoid unnecessary recreation
        import hashlib
        import json
        cache_key_data = {
            "model_id": self.model_id,
            "region_name": self.region_name,
            "model_kwargs": model_kwargs,
            "endpoint_url": self.endpoint_url,
            "streaming": self.stream,
        }
        cache_key = hashlib.md5(json.dumps(cache_key_data, sort_keys=True, default=str).encode()).hexdigest()

        # Return cached model if parameters haven't changed
        if self._cached_model is not None and self._cached_model_key == cache_key:
            return self._cached_model
        
        try:
            output = ChatBedrock(
                client=boto3_client,
                model_id=self.model_id,
                region_name=self.region_name,
                model_kwargs=model_kwargs,
                endpoint_url=self.endpoint_url,
                streaming=self.stream,
            )
            # Cache the model instance and key
            self._cached_model = output
            self._cached_model_key = cache_key
        except Exception as e:
            msg = f"Could not connect to AmazonBedrock API: {str(e)}"
            raise ValueError(msg) from e

        return output
