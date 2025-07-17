import pytest

from mcpd_sdk.exceptions import McpdError


class TestMcpdError:
    def test_mcpd_error_inheritance(self):
        """Test that McpdError inherits from Exception."""
        assert issubclass(McpdError, Exception)

    def test_mcpd_error_basic_creation(self):
        """Test basic McpdError creation."""
        error = McpdError("Test error message")
        assert str(error) == "Test error message"

    def test_mcpd_error_empty_message(self):
        """Test McpdError with empty message."""
        error = McpdError("")
        assert str(error) == ""

    def test_mcpd_error_none_message(self):
        """Test McpdError with None message."""
        error = McpdError(None)
        assert str(error) == "None"

    def test_mcpd_error_with_args(self):
        """Test McpdError with multiple arguments."""
        error = McpdError("Error", "with", "multiple", "args")
        assert "Error" in str(error)

    def test_mcpd_error_raising(self):
        """Test that McpdError can be raised and caught."""
        with pytest.raises(McpdError):
            raise McpdError("Test error")

    def test_mcpd_error_catching_as_exception(self):
        """Test that McpdError can be caught as Exception."""
        with pytest.raises(Exception):
            raise McpdError("Test error")

    def test_mcpd_error_chaining(self):
        """Test error chaining with McpdError."""
        original_error = ValueError("Original error")
        
        try:
            raise original_error
        except ValueError as e:
            chained_error = McpdError("Chained error")
            chained_error.__cause__ = e
            
        assert chained_error.__cause__ is original_error
        assert str(chained_error) == "Chained error"

    def test_mcpd_error_with_format_string(self):
        """Test McpdError with format string."""
        server_name = "test_server"
        tool_name = "test_tool"
        error = McpdError(f"Error calling tool '{tool_name}' on server '{server_name}'")
        
        expected_message = "Error calling tool 'test_tool' on server 'test_server'"
        assert str(error) == expected_message

    def test_mcpd_error_attributes(self):
        """Test that McpdError has expected attributes."""
        error = McpdError("Test error")
        
        # Should have standard Exception attributes
        assert hasattr(error, 'args')
        assert error.args == ("Test error",)

    def test_mcpd_error_repr(self):
        """Test string representation of McpdError."""
        error = McpdError("Test error")
        repr_str = repr(error)
        
        assert "McpdError" in repr_str
        assert "Test error" in repr_str

    def test_mcpd_error_instance_check(self):
        """Test isinstance checks with McpdError."""
        error = McpdError("Test error")
        
        assert isinstance(error, McpdError)
        assert isinstance(error, Exception)
        assert isinstance(error, BaseException)

    def test_mcpd_error_equality(self):
        """Test equality comparison of McpdError instances."""
        error1 = McpdError("Same message")
        error2 = McpdError("Same message")
        error3 = McpdError("Different message")
        
        # Note: Exception instances are not equal even with same message
        # This is standard Python behavior
        assert error1 is not error2
        assert error1 is not error3

    def test_mcpd_error_with_complex_message(self):
        """Test McpdError with complex message containing various data types."""
        data = {"server": "test", "tool": "example", "params": [1, 2, 3]}
        error = McpdError(f"Complex error with data: {data}")
        
        assert "Complex error with data:" in str(error)
        assert "test" in str(error)
        assert "example" in str(error)