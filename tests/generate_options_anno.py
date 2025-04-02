import toml
from pydantic import BaseModel, Field
from typing import Optional

class GAEOptions(BaseModel):
    learning_rate: float
    alpha: float
    dim: int  # Nested field

class ExampleModel(BaseModel):
    name: str
    age: Optional[int] = Field(None, ge=18, description="The age must be at least 18")
    city: str = "Unknown"
    model_configuration: GAEOptions  # Nested model

def model_to_toml(model: BaseModel) -> str:
    toml_data = {}

    # Iterate over the fields of the model itself (not nested models)
    for field, value in model.__annotations__.items():
        # Handle nested models (e.g., model_configuration) separately
        if isinstance(getattr(model, field, None), BaseModel):
            # Recursively handle nested model
            nested_model = getattr(model, field)
            toml_data[field] = model_to_toml(nested_model)  # Recursively convert nested model to TOML
        else:
            # Handle regular fields
            field_value = getattr(model, field, None)
            field_info = model.model_fields.get(field)
            
            if field_info:
                field_metadata = {}
                if field_info.default is not None:
                    field_metadata['default'] = field_info.default
                if field_info.description:
                    field_metadata['description'] = field_info.description
                
                # Add field data to the TOML dictionary
                toml_data[field] = field_metadata

    # Convert the dictionary to TOML format string
    return toml.dumps({k: v for k, v in toml_data.items()})

# Test
gae_config = GAEOptions(learning_rate=0.01, alpha=0.5, dim=128)
example_config = ExampleModel(name="John", age=25, city="New York", model_configuration=gae_config)

# Convert the ExampleModel instance to TOML format
toml_str = model_to_toml(example_config)
print(toml_str)

