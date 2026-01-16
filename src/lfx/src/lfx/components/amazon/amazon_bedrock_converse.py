from langflow.field_typing import LanguageModel
from langflow.inputs.inputs import BoolInput, FloatInput, IntInput, MessageTextInput, SecretStrInput
from langflow.io import DictInput, DropdownInput

from lfx.base.models.aws_constants import AWS_REGIONS, AWS_MODEL_IDs
from lfx.base.models.model import LCModelComponent
from lfx.field_typing.range_spec import RangeSpec


class AmazonBedrockConverseComponent(LCModelComponent):
    display_name: str = "ANSWER_AWS"
    description: str = (
        "Generate text using Amazon Bedrock LLMs with the modern Converse API for improved conversation handling."
    )
    icon = "Amazon"
    name = "AmazonBedrockConverseModel"
    beta = True

    def __init__(self, **kwargs):
        """Initialize the component and create caches for boto3 session and client."""
        super().__init__(**kwargs)
        self._cached_boto3_session = None
        self._cached_boto3_client = None

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
            info="Maximum number of tokens to generate. Set to 0 for unlimited tokens.",
            advanced=False,
            range_spec=RangeSpec(min=0, max=128000),
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
        BoolInput(
            name="disable_streaming",
            display_name="Disable Streaming",
            value=False,
            info="If True, disables streaming responses. Useful for batch processing.",
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
            from langchain_aws.chat_models.bedrock_converse import ChatBedrockConverse
        except ImportError as e:
            msg = "langchain_aws is not installed. Please install it with `pip install langchain_aws`."
            raise ImportError(msg) from e
        try:
            import boto3
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

        client_params = {}
        if self.endpoint_url:
            client_params["endpoint_url"] = self.endpoint_url
        if self.region_name:
            client_params["region_name"] = self.region_name

        # Cache boto3 client to avoid creating multiple clients and prevent file handle leaks
        if self._cached_boto3_client is None:
            self._cached_boto3_client = session.client("bedrock-runtime", **client_params)

        boto3_client = self._cached_boto3_client

        # Prepare initialization parameters
        init_params = {
            "model": self.model_id,
            "client": boto3_client,
        }

        # Note: When passing a client, region_name and credentials are handled by the client
        # Only add endpoint_url if not already in client_params
        if self.endpoint_url:
            init_params["endpoint_url"] = self.endpoint_url

        # Add model parameters - only send one of temperature or top_p based on sampling_strategy
        sampling_strategy = getattr(self, "sampling_strategy", "temperature")
        
        if sampling_strategy == "top_p":
            if hasattr(self, "top_p") and self.top_p is not None:
                init_params["top_p"] = self.top_p
        else:
            # Default to temperature if not specified or if sampling_strategy is "temperature"
            if hasattr(self, "temperature") and self.temperature is not None:
                init_params["temperature"] = self.temperature
        
        if hasattr(self, "max_tokens") and self.max_tokens is not None:
            init_params["max_tokens"] = self.max_tokens

        # Handle streaming - only disable if explicitly requested
        if hasattr(self, "disable_streaming") and self.disable_streaming:
            init_params["disable_streaming"] = True

        # Handle additional model request fields carefully
        # Based on the error, inferenceConfig should not be passed as additional fields for some models
        additional_model_request_fields = {}

        # Only add top_k if user explicitly provided additional fields or if needed for specific models
        if hasattr(self, "additional_model_fields") and self.additional_model_fields:
            for field in self.additional_model_fields:
                if isinstance(field, dict):
                    additional_model_request_fields.update(field)

        # For now, don't automatically add inferenceConfig for top_k to avoid validation errors
        # Users can manually add it via additional_model_fields if their model supports it

        # Only add if we have actual additional fields
        if additional_model_request_fields:
            init_params["additional_model_request_fields"] = additional_model_request_fields

        try:
            output = ChatBedrockConverse(**init_params)
        except Exception as e:
            # Provide helpful error message with fallback suggestions
            error_details = str(e)
            if "validation error" in error_details.lower():
                msg = (
                    f"ChatBedrockConverse validation error: {error_details}. "
                    f"This may be due to incompatible parameters for model '{self.model_id}'. "
                    f"Consider adjusting the model parameters or trying the legacy Amazon Bedrock component."
                )
            elif "converse api" in error_details.lower():
                msg = (
                    f"Converse API error: {error_details}. "
                    f"The model '{self.model_id}' may not support the Converse API. "
                    f"Try using the legacy Amazon Bedrock component instead."
                )
            else:
                msg = f"Could not initialize ChatBedrockConverse: {error_details}"
            raise ValueError(msg) from e

        return output
